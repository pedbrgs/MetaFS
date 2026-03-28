import gc
import os
import time
import random
import logging
import argparse
import multiprocessing as mp
import pandas as pd

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

from pyccea.coevolution import CCPSTFG

MAX_FS_HOURS = 12
MAX_FS_SECONDS = MAX_FS_HOURS * 3600


class FeatureSelectionTimeout(BaseException):
    pass


from pyccea.utils.datasets import DataLoader
from pyccea.evaluation.wrapper import WrapperEvaluation


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


def list_datasets(data_dir: str) -> list:
    data_stats = []
    for file in os.listdir(data_dir):
        if file.endswith(".parquet"):
            data = pd.read_parquet(os.path.join(data_dir, file))
            num_samples, num_features = data.shape
            dataset_name = file.replace(".parquet", "")
            del data
            gc.collect()
            data_stats.append({
                "data_path": dataset_name,
                "num_samples": num_samples,
                "num_features": num_features 
            })
    data = pd.DataFrame(data_stats)
    data["computational_effort"] = data["num_samples"] + data["num_features"]
    return data.sort_values("computational_effort", ascending=False)["data_path"].values.tolist()


def set_logger() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers = []
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)


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


def load_ccea_conf(random_state: int, is_debug: bool, n_workers: int = 1) -> dict:
    return {
        "coevolution": {
            "subpop_sizes": [50],
            "max_gen": 1 if is_debug else 1000,
            "max_gen_without_improvement": 2 if is_debug else 10,
            "optimized_resource_allocation": True,
            "max_best_context_vectors": 0,
            "seed" : random_state
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
            "use_subprocess": True,
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


def get_completed_datasets(
        results: pd.DataFrame,
        metric_col: str,
        standard_error_threshold: float,
        min_runs: int,
        max_runs: int
    ) -> list:
    errors = results.groupby("dataset")[metric_col].apply(cumulative_standard_error)
    achieved_errors = errors.groupby(level=0).last()
    run_counts = results.groupby("dataset").size()
    # Condition 1: achieved error threshold and meets minimum number of runs
    cond_error_met = (achieved_errors <= standard_error_threshold) & (run_counts >= min_runs)
    # Condition 2: did not achieve error threshold but reached maximum number of runs
    cond_error_failed = (achieved_errors > standard_error_threshold) & (run_counts >= max_runs)
    # Completed if either condition holds
    completed_mask = cond_error_met | cond_error_failed
    return achieved_errors.index[completed_mask].tolist()


def evaluate_context_vector(ccea, subset: str) -> pd.DataFrame:
    evaluator = WrapperEvaluation(
        task=ccea.conf["wrapper"]["task"],
        model_type=ccea.conf["wrapper"]["model_type"],
        eval_function=ccea.conf["evaluation"]["eval_function"],
        eval_mode="k_fold" if subset == "train" else "hold_out",
        n_classes=ccea.data.n_classes
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
        "quantile_to_remove": kwargs["ccea"].quantile_to_remove,
        "n_pls_components": kwargs["ccea"].n_components,
        "vip_threshold": kwargs["ccea"].vip_threshold,
        "removed_features": str(kwargs["ccea"].removed_features),
        "n_iterations": len(kwargs["ccea"].convergence_curve),
        "n_selected_features": kwargs["ccea"].best_context_vector.sum(),
        "n_pre_removed_features": len(kwargs["ccea"].removed_features),
        "init_time": kwargs["init_time"],
        "tuning_time": kwargs["ccea"]._tuning_time,
        "feature_selection_time": kwargs["fs_time"]
    }
    return pd.DataFrame.from_dict(run_stats, orient="index").T


def _ccea_worker(result_queue, data_path, dataset_name, data_conf, ccea_conf, run_num):
    """Run CCEA init + optimize + evaluation in an isolated process.

    Calls os.setpgrp() so the entire process group (including n_workers subprocesses
    spawned by CCEA) can be killed atomically with os.killpg() on timeout.
    """
    os.setpgrp()
    try:
        dataloader = build_dataloader(
            data_path=data_path,
            dataset_name=dataset_name,
            data_conf=data_conf,
        )
        dataloader.get_ready()

        t0 = time.time()
        ccea = CCPSTFG(conf=ccea_conf, data=dataloader, verbose=False)
        init_time = time.time() - t0

        t1 = time.time()
        ccea.optimize()
        fs_time = time.time() - t1

        train_metrics = evaluate_context_vector(ccea, subset="train")
        test_metrics = evaluate_context_vector(ccea, subset="test")
        run_stats = get_overall_stats(
            dataset_name=dataset_name,
            ccea=ccea,
            run=run_num,
            init_time=init_time,
            fs_time=fs_time,
        )
        result_queue.put(("ok", run_stats, train_metrics, test_metrics, init_time, fs_time))
    except Exception as e:
        result_queue.put(("error", str(e)))


def check_stopping_criteria(results: pd.DataFrame, args: dict, dataset_name: str, n_runs: int) -> bool:
    metric_series = results[results["dataset"] == dataset_name][args.metric_col].dropna()
    if metric_series.empty:
        return False
    errors = cumulative_standard_error(metric_series)
    if (n_runs >= args.min_runs) and (errors.iloc[-1] <= args.se_thresh):
        logging.info(f"Standard error threshold achieved ({errors.iloc[-1]:.2f}%). ")
        logging.info(f"Ending experiments for dataset: {dataset_name}.")
        return True
    if n_runs >= args.max_runs:
        logging.info(f"Maximum number of runs reached ({args.max_runs}). ")
        logging.info(f"Ending experiments for dataset: {dataset_name}.")
        return True
    return False


def run(args: dict) -> None:

    set_logger()
    args = parse_args()

    datasets = list_datasets(data_dir=args.data_dir)
    if args.is_debug:
        datasets = datasets[0:3]
    logging.info(f"Datasets: {datasets}.")
    results = load_results()
    if not results.empty:
        completed_datasets = get_completed_datasets(
            results=results,
            metric_col=args.metric_col,
            standard_error_threshold=args.se_thresh,
            min_runs=args.min_runs,
            max_runs=args.max_runs
        )
        datasets = [dataset for dataset in datasets if dataset not in completed_datasets]
        logging.info(f"Completed datasets: {completed_datasets}.")
    logging.info(f"Datasets for experiments: {datasets}.")

    for dataset_name in datasets:

        dataset_file = f"{dataset_name}.parquet"
        data_path = os.path.join(args.data_dir, dataset_file)
        logging.info(f"Starting experiments for dataset: {dataset_name}.")

        n_runs = (
            results.loc[results["dataset"] == dataset_name, "run"].max()
            if not results[results["dataset"] == dataset_name].empty
            else 0
        )
        while True:

            n_runs += 1
            random_state = random.randint(0, 10_000)
            logging.info(f"Run #{n_runs} | Random state {random_state}")

            # Load data and CCEA configuration (dataloader is built inside the worker)
            data_conf = load_data_conf(random_state=random_state)
            # Load CCEA configuration
            ccea_conf = load_ccea_conf(random_state=random_state, is_debug=args.is_debug, n_workers=args.n_workers)

            result_queue = mp.Queue()
            proc = mp.Process(
                target=_ccea_worker,
                args=(result_queue, data_path, dataset_name, data_conf, ccea_conf, n_runs),
            )
            start_time = time.time()
            proc.start()
            proc.join(timeout=MAX_FS_SECONDS)

            if proc.is_alive():
                fs_time = round(time.time() - start_time, 2)
                try:
                    os.killpg(proc.pid, 9)  # SIGKILL to entire process group
                except ProcessLookupError:
                    pass
                proc.join()
                logging.warning(f"Run #{n_runs} exceeded {MAX_FS_HOURS}h time limit.")
                run_results = pd.DataFrame([{
                    "dataset": dataset_name,
                    "run": n_runs,
                    "feature_selection_time": fs_time,
                }])
                results = pd.concat([results, run_results], ignore_index=True)
                save_results(results=results)
                break

            outcome = result_queue.get() if not result_queue.empty() else ("error", "Worker produced no result")
            if outcome[0] == "error":
                logging.error(f"Feature selection failed: {outcome[1]}")
                break
            _, run_stats, train_metrics, test_metrics, init_time, fs_time = outcome
            logging.info(f"CCEA initialization completed in {(init_time/60):.2f} minutes.")
            logging.info(f"Feature selection completed in {(fs_time/60):.2f} minutes.")

            gc.collect()

            # Aggregate and save results
            run_results = pd.concat([run_stats, train_metrics, test_metrics], axis=1)
            results = pd.concat([results, run_results], ignore_index=True)
            save_results(results=results)
            # Check stopping criteria
            if check_stopping_criteria(results=results, args=args, dataset_name=dataset_name, n_runs=n_runs):
                break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, help="data directory path")
    parser.add_argument("--metric-col", type=str, help="metric to be monitored for standard error threshold")
    parser.add_argument("--se-thresh", type=float, default=0.03, help="standard error threshold")
    parser.add_argument("--min-runs", type=int, default=5, help="minimum number of runs per dataset")
    parser.add_argument("--max-runs", type=int, default=50, help="maximum number of runs per dataset")
    parser.add_argument("--n-workers", type=int, default=1, help="number of parallel workers for CCEA evaluation (default: 1)")
    parser.add_argument("--is-debug", action="store_true", help="debug mode")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
