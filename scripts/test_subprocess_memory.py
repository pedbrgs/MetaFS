import gc
import os
import time
import random
import logging
import argparse
import pandas as pd
import psutil  # pip install psutil

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

from pyccea.coevolution import CCPSTFG
from pyccea.utils.datasets import DataLoader
from pyccea.evaluation.wrapper import WrapperEvaluation


def _sample_rss_mb() -> float:
    """Return current RSS of this process in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


class MemoryTrackedCCPSTFG(CCPSTFG):
    """CCPSTFG subclass that records RSS after each generation's evaluation phase.

    Adds two attributes after optimize() completes:
        memory_curve : list[float]
            RSS in MB sampled once per generation, right after
            _evaluate_evolved_subpopulations returns.  Index aligns with
            convergence_curve (same length at the end of the run).
        init_memory_mb : float
            RSS right after subpopulation initialisation finishes.
    """

    def _init_subpopulations(self):
        super()._init_subpopulations()
        self.init_memory_mb = _sample_rss_mb()

    def _evaluate_evolved_subpopulations(self, *args, **kwargs):
        result = super()._evaluate_evolved_subpopulations(*args, **kwargs)
        self._memory_curve.append(_sample_rss_mb())
        return result

    def optimize(self):
        self._memory_curve = []
        super().optimize()
        # Expose as public attribute with rounded values (matches convergence_curve style)
        self.memory_curve = [round(v, 2) for v in self._memory_curve]


def build_dataloader(data_path: str, dataset_name: str, data_conf: dict) -> DataLoader:
    DataLoader.DATASETS[dataset_name] = {
        "file": data_path,
        "task": "classification"
    }
    dataloader = DataLoader(
        dataset=dataset_name,
        conf=data_conf
    )
    return dataloader


logger = logging.getLogger("CCEA_Experiment")

def set_logger() -> None:
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

set_logger()


def load_results(root_path: str = "results", output_file: str = "experiments.parquet") -> pd.DataFrame:
    os.makedirs(root_path, exist_ok=True)
    file_path = os.path.join(root_path, output_file)
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    else:
        return pd.DataFrame(columns=["dataset"])


def save_results(results: pd.DataFrame, root_path: str = "results", output_file: str = "experiments.parquet") -> None:
    os.makedirs(root_path, exist_ok=True)
    file_path = os.path.join(root_path, output_file)
    results.to_parquet(file_path, index=False)


def load_data_conf(random_state: int) -> dict:
    return {
        "general": {
            "splitter_type": "k_fold",
            "verbose": True,
            "float_dtype": "float32",
            "seed": random_state
        },
        "splitter": {
            "preset": True,
            "kfolds": 3,
            "prefold": False
        },
        "normalization": {
            "normalize": True,
            "method": "standard"
        },
        "preprocessing": {
            "drop_na": True,
            "winsorization": True,
            "quantiles": [0.01, 0.99]
        }
    }


def load_ccea_conf(random_state: int, is_debug: bool, n_workers: int, use_subprocess: bool) -> dict:
    return {
        "coevolution": {
            "subpop_sizes": [50],
            "max_gen": 1 if is_debug else 1000,
            "max_gen_without_improvement": 2 if is_debug else 10,
            "optimized_resource_allocation": True,
            "max_best_context_vectors": 0,
            "seed": random_state
        },
        "decomposition": {
            "method": "clustering",
            "drop": True,
            "max_n_clusters": 10,
            "max_n_pls_components": 10,
            "removal_quantile_step_size": 0.05,
            "max_removal_quantile": 0.95,
            "clustering_model_type": "agglomerative_clustering"
        },
        "collaboration": {
            "method": "best"
        },
        "wrapper": {
            "task": "classification",
            "cache_size": 2000,
            "model_type": "random_forest",
            "use_subprocess": use_subprocess,
        },
        "evaluation": {
            "fitness_function": "penalty",
            "eval_function": "balanced_accuracy",
            "eval_mode": "k_fold",
            "weights": [1.0, 0.0],
            "n_workers": n_workers
        },
        "optimizer": {
            "method": "GA",
            "selection_method": "generational",
            "mutation_rate": 0.05,
            "crossover_rate": 1.0,
            "tournament_sample_size": 1,
            "elite_size": 1
        }
    }


def cumulative_standard_error(series: pd.Series) -> pd.Series:
    return series.expanding().std() / series.expanding().count().pow(0.5)


def get_completed_tasks(results: pd.DataFrame, args) -> set:
    if results.empty or "use_subprocess" not in results.columns:
        return set()

    def check_group(group):
        errors = cumulative_standard_error(group[args.metric_col])
        last_error = errors.iloc[-1]
        n_runs = len(group)
        cond_met = (last_error <= args.se_thresh) and (n_runs >= args.min_runs)
        cond_max = (n_runs >= args.max_runs)
        return cond_met or cond_max

    status = results.groupby(["dataset", "use_subprocess"]).apply(check_group)
    completed = status[status == True].index.tolist()
    return set(completed)


def evaluate_context_vector(ccea, subset: str) -> pd.DataFrame:
    evaluator = WrapperEvaluation(
        task=ccea.conf["wrapper"]["task"],
        model_type=ccea.conf["wrapper"]["model_type"],
        eval_function=ccea.conf["evaluation"]["eval_function"],
        eval_mode="k_fold" if subset == "train" else "hold_out",
        n_classes=ccea.data.n_classes,
        store_estimators=False,
        use_subprocess=False  # Final evaluation always in-process
    )
    _ = evaluator.evaluate(
        solution=ccea.best_context_vector.copy(),
        data=ccea.data
    )
    metrics = pd.DataFrame.from_dict(
        evaluator.evaluations,
        orient="index"
    ).transpose()
    metrics.columns = [f"{subset}_{col}" for col in metrics.columns]
    del evaluator
    gc.collect()
    return metrics


def get_overall_stats(**kwargs) -> dict:
    run_stats = {
        "dataset": kwargs["dataset_name"],
        "n_workers": kwargs["n_workers"],
        "use_subprocess": kwargs["use_subprocess"],
        "total_samples": kwargs["ccea"].data.n_examples,
        "total_features": kwargs["ccea"].data.n_features,
        "run": kwargs["run"],
        "n_subcomps": kwargs["ccea"].n_subcomps,
        "subcomp_sizes": str(kwargs["ccea"].subcomp_sizes),
        "subpop_sizes": str(kwargs["ccea"].subpop_sizes),
        "ccea_conf": str(kwargs["ccea"].conf),
        "data_conf": str(kwargs["ccea"].data.conf),
        "feature_indices": str(kwargs["ccea"].best_feature_idxs),
        "best_context_vector": str(kwargs["ccea"].best_context_vector),
        "best_fitness": round(kwargs["ccea"].best_fitness, 4),
        "convergence_curve": [round(fitness, 4) for fitness in kwargs["ccea"].convergence_curve],
        "memory_curve": kwargs["ccea"].memory_curve,
        "init_memory_mb": round(getattr(kwargs["ccea"], "init_memory_mb", 0.0), 2),
        "peak_memory_mb": max(kwargs["ccea"].memory_curve) if kwargs["ccea"].memory_curve else 0.0,
        "quantile_to_remove": kwargs["ccea"].quantile_to_remove,
        "n_pls_components": kwargs["ccea"].n_components,
        "vip_threshold": kwargs["ccea"].vip_threshold,
        "removed_features": str(kwargs["ccea"].removed_features),
        "n_iterations": len(kwargs["ccea"].convergence_curve),
        "n_selected_features": kwargs["ccea"].best_context_vector.sum(),
        "n_pre_removed_features": len(kwargs["ccea"].removed_features),
        "init_time": kwargs["init_time"],
        "tuning_time": kwargs["ccea"]._tuning_time,
        "feature_selection_time": kwargs["fs_time"],
    }
    return pd.DataFrame.from_dict(run_stats, orient="index").T


def check_stopping_criteria(
    results: pd.DataFrame, args, dataset_name: str, n_runs: int, use_subprocess: bool
) -> bool:
    mask = (results["dataset"] == dataset_name) & (results["use_subprocess"] == use_subprocess)
    metric_series = results[mask][args.metric_col]

    if metric_series.empty:
        return False

    errors = cumulative_standard_error(metric_series)
    last_error = errors.iloc[-1]

    if (n_runs >= args.min_runs) and (last_error <= args.se_thresh):
        logger.info(
            f"Stopping threshold reached for {dataset_name} "
            f"(use_subprocess={use_subprocess}): {last_error:.4f}"
        )
        return True
    if n_runs >= args.max_runs:
        logger.info(
            f"Maximum number of runs reached for {dataset_name} "
            f"(use_subprocess={use_subprocess})"
        )
        return True
    return False


def run(args) -> None:
    set_logger()

    if os.name != "posix":
        logger.warning(
            "WARNING: use_subprocess relies on os.fork() and only works on POSIX systems. "
            "Results for use_subprocess=True will be identical to use_subprocess=False on this platform."
        )

    datasets = ["swarm_behaviour_aligned", "pcam"]
    subprocess_flags = [False, True]

    logger.info(f"Datasets: {datasets}")
    logger.info(f"n_workers: {args.n_workers}")
    logger.info(f"subprocess flags: {subprocess_flags}")

    results = load_results(output_file="memory_subprocess_results.parquet")
    completed_tasks = get_completed_tasks(results, args)

    for dataset_name in datasets:

        dataset_file = f"{dataset_name}.parquet"
        data_path = os.path.join(args.data_dir, dataset_file)
        logger.info(f"Starting experiments for dataset: {dataset_name}.")

        for use_subprocess in subprocess_flags:

            if (dataset_name, use_subprocess) in completed_tasks:
                logger.info(
                    f"Skipping {dataset_name} (use_subprocess={use_subprocess}): "
                    "criterion already satisfied."
                )
                continue

            logger.info(f"Experiment: {dataset_name} | use_subprocess={use_subprocess}")

            if not results.empty and "use_subprocess" in results.columns:
                mask = (
                    (results["dataset"] == dataset_name) &
                    (results["use_subprocess"] == use_subprocess)
                )
                n_runs = int(results[mask]["run"].max()) if not results[mask].empty else 0
            else:
                n_runs = 0

            while True:
                n_runs += 1
                random_state = random.randint(0, 10_000)
                logger.info(
                    f"Run #{n_runs} for {dataset_name} (use_subprocess={use_subprocess})"
                )

                data_conf = load_data_conf(random_state=random_state)
                dataloader = build_dataloader(data_path, dataset_name, data_conf)
                dataloader.get_ready()

                ccea_conf = load_ccea_conf(
                    random_state=random_state,
                    is_debug=args.is_debug,
                    n_workers=args.n_workers,
                    use_subprocess=use_subprocess
                )

                start_init = time.time()
                ccea = MemoryTrackedCCPSTFG(conf=ccea_conf, data=dataloader, verbose=False)
                init_time = time.time() - start_init

                start_fs = time.time()
                ccea.optimize()
                fs_time = time.time() - start_fs

                peak_memory_mb = max(ccea.memory_curve) if ccea.memory_curve else 0.0
                logger.info(f"Peak RSS: {peak_memory_mb:.1f} MB | FS time: {fs_time/3600:.4f} h")

                train_metrics = evaluate_context_vector(ccea, subset="train")
                test_metrics = evaluate_context_vector(ccea, subset="test")

                run_stats = get_overall_stats(
                    dataset_name=dataset_name,
                    n_workers=args.n_workers,
                    use_subprocess=use_subprocess,
                    ccea=ccea,
                    run=n_runs,
                    init_time=init_time,
                    fs_time=fs_time,
                )

                run_results = pd.concat([run_stats, train_metrics, test_metrics], axis=1)
                results = pd.concat([results, run_results], ignore_index=True)
                save_results(results, output_file="memory_subprocess_results.parquet")

                del dataloader, ccea
                gc.collect()

                if check_stopping_criteria(results, args, dataset_name, n_runs, use_subprocess):
                    break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--n-workers", type=int, default=4,
                        help="Number of parallel evaluation workers (fixed for this experiment).")
    parser.add_argument("--metric-col", type=str, default="test_balanced_accuracy")
    parser.add_argument("--se-thresh", type=float, default=0.03)
    parser.add_argument("--min-runs", type=int, default=5)
    parser.add_argument("--max-runs", type=int, default=30)
    parser.add_argument("--is-debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
