import gc
import os
import time
import random
import logging
import argparse
import warnings
import multiprocessing as mp
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score

warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

from pyccea.utils.datasets import DataLoader

MAX_FS_HOURS = 12
MAX_FS_SECONDS = MAX_FS_HOURS * 3600


def _ga_worker(
    result_queue,
    X_train, y_train,
    n_features, random_state, n_jobs, kfolds,
    population_size, generations, crossover_prob, mutation_prob, patience,
):
    """Run genetic algorithm feature selection in an isolated process.

    Calls os.setpgrp() so the entire process group (including joblib workers)
    can be killed atomically with os.killpg() on timeout.
    """
    if hasattr(os, "setpgrp"):
        os.setpgrp()
    try:
        import random as _random
        _random.seed(random_state)
        np.random.seed(random_state)

        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMax)

        estimator = RandomForestClassifier(random_state=random_state, class_weight="balanced")

        def evaluate(individual):
            selected = [i for i, bit in enumerate(individual) if bit == 1]
            if len(selected) == 0:
                return (0.0,)
            scores = cross_val_score(
                estimator, X_train[:, selected], y_train,
                cv=kfolds, scoring="balanced_accuracy", n_jobs=1
            )
            return (float(scores.mean()),)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", _random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", evaluate)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / n_features)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=population_size)
        hof = tools.HallOfFame(1)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        stats.register("avg", np.mean)

        history = {"fitness_max": [], "fitness_avg": []}

        # Evaluate initial population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        hof.update(pop)

        best_fitness = hof[0].fitness.values[0]
        no_improvement = 0

        t0 = time.time()

        for gen in range(generations):
            offspring = algorithms.varOr(pop, toolbox, population_size, crossover_prob, mutation_prob)

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            pop = toolbox.select(pop + offspring, population_size)
            hof.update(pop)

            record = stats.compile(pop)
            history["fitness_max"].append(float(record["max"]))
            history["fitness_avg"].append(float(record["avg"]))
            logging.info(f"Gen {gen + 1}/{generations} | max={record['max']:.4f} avg={record['avg']:.4f}")

            current_best = hof[0].fitness.values[0]
            if current_best > best_fitness:
                best_fitness = current_best
                no_improvement = 0
            else:
                no_improvement += 1

            if patience is not None and no_improvement >= patience:
                logging.info(f"Early stopping at generation {gen + 1} ({patience} generations without improvement).")
                break

        fit_time = time.time() - t0

        selected_features = [f"feat_{i}" for i, bit in enumerate(hof[0]) if bit == 1]
        result_queue.put(("ok", selected_features, best_fitness, history, fit_time))
    except Exception as e:
        import traceback
        result_queue.put(("error", traceback.format_exc()))


def compute_evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray, subset_name: str) -> dict:
    """Computes a comprehensive suite of binary classification metrics for a given subset.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth label.
    y_pred : np.ndarray
        Predicted output.
    subset_name : str
        Subset name.

    Returns
    -------
    metrics_dict : dict
        Classification metrics.
    """
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics_dict = {
        f"{subset_name}_accuracy": accuracy_score(y_true, y_pred),
        f"{subset_name}_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"{subset_name}_precision": precision_score(y_true, y_pred, average="binary"),
        f"{subset_name}_recall": recall_score(y_true, y_pred, average="binary"),
        f"{subset_name}_f1_score": f1_score(y_true, y_pred, average="binary"),
        f"{subset_name}_specificity": specificity
    }
    return metrics_dict


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
    return data.sort_values("computational_effort", ascending=True)["data_path"].values.tolist()


def set_logger() -> None:
    logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers = []
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)


def load_results(root_path: str = "results", output_file: str = None) -> pd.DataFrame:
    os.makedirs(root_path, exist_ok=True)
    file_path = os.path.join(root_path, output_file)
    if os.path.exists(file_path):
        return pd.read_parquet(file_path)
    else:
        return pd.DataFrame(columns=["dataset"])


def save_results(results: pd.DataFrame, root_path: str = "results", output_file: str = None) -> None:
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


def cumulative_standard_error(series: pd.Series) -> pd.Series:
    return series.expanding().std() / series.expanding().count().pow(0.5)


def get_completed_datasets(
        results: pd.DataFrame,
        metric_col: str,
        standard_error_threshold: float,
        min_runs: int,
        max_runs: int
    ) -> list:
    if metric_col not in results.columns:
        timed_out = results.groupby("dataset").size()
        return timed_out.index.tolist()
    timed_out_datasets = (
        results.groupby("dataset")[metric_col]
        .apply(lambda s: s.isna().any())
    )
    errors = results.groupby("dataset")[metric_col].apply(cumulative_standard_error)
    achieved_errors = errors.groupby(level=0).last()
    run_counts = results.groupby("dataset").size()
    cond_error_met = (achieved_errors <= standard_error_threshold) & (run_counts >= min_runs)
    cond_error_failed = (achieved_errors > standard_error_threshold) & (run_counts >= max_runs)
    completed_mask = cond_error_met | cond_error_failed | timed_out_datasets
    return achieved_errors.index[completed_mask].tolist()


def check_stopping_criteria(results: pd.DataFrame, args, dataset_name: str, n_runs: int) -> bool:
    metric_series = results[results["dataset"] == dataset_name][args.metric_col].dropna()
    if metric_series.empty:
        return False
    errors = cumulative_standard_error(metric_series)
    if (n_runs >= args.min_runs) and (errors.iloc[-1] <= args.se_thresh):
        logging.info(f"Standard error threshold achieved ({errors.iloc[-1]:.4f}).")
        logging.info(f"Ending experiments for dataset: {dataset_name}.")
        return True
    if n_runs >= args.max_runs:
        logging.info(f"Maximum number of runs reached ({args.max_runs}).")
        logging.info(f"Ending experiments for dataset: {dataset_name}.")
        return True
    return False


def run(args) -> None:

    set_logger()

    output_file = "ga_experiments.parquet"

    datasets = list_datasets(data_dir=args.data_dir)
    if args.is_debug:
        datasets = datasets[0:3]
    logging.info(f"Datasets: {datasets}.")

    results = load_results(output_file=output_file)
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

            # Load data
            data_conf = load_data_conf(random_state=random_state)
            dataloader = build_dataloader(
                data_path=data_path,
                dataset_name=dataset_name,
                data_conf=data_conf
            )
            dataloader.get_ready()

            feature_names = [f"feat_{i}" for i in range(dataloader.n_features)]

            result_queue = mp.Queue()
            proc = mp.Process(
                target=_ga_worker,
                args=(
                    result_queue,
                    dataloader.X_train, dataloader.y_train,
                    dataloader.n_features, random_state, args.n_jobs, 3,
                    args.population_size,
                    2 if args.is_debug else args.generations,
                    args.crossover_prob,
                    args.mutation_prob,
                    args.patience,
                ),
            )
            start_time = time.time()
            proc.start()
            proc.join(timeout=MAX_FS_SECONDS)

            if proc.is_alive():
                elapsed = round(time.time() - start_time, 2)
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(proc.pid, 9)
                    else:
                        proc.kill()
                except ProcessLookupError:
                    pass
                proc.join()
                logging.warning(f"Run #{n_runs} exceeded {MAX_FS_HOURS}h time limit.")
                run_stats = {
                    "dataset": dataset_name,
                    "run": n_runs,
                    "method": "ga",
                    "fs_time": elapsed,
                    "n_selected_features": None,
                    "best_cv_score": None,
                    "ga_fitness_max": None,
                    "ga_fitness_avg": None,
                }
                results = pd.concat([results, pd.DataFrame([run_stats])], ignore_index=True)
                save_results(results=results, output_file=output_file)
                break

            outcome = result_queue.get() if not result_queue.empty() else ("error", "Worker produced no result")
            if outcome[0] == "error":
                logging.error(f"Genetic algorithm failed: {outcome[1]}")
                break
            _, selected_features, best_cv_score, history, fit_time = outcome
            logging.info(f"{len(selected_features)} features selected by GA (best CV score: {best_cv_score:.4f}).")

            selected_indices = [feature_names.index(f) for f in selected_features]

            # Train final model on selected features
            final_model = RandomForestClassifier(random_state=random_state, class_weight="balanced")
            final_model.fit(dataloader.X_train[:, selected_indices], dataloader.y_train)

            # Evaluate
            train_metrics = compute_evaluation_metrics(
                y_true=dataloader.y_train,
                y_pred=final_model.predict(dataloader.X_train[:, selected_indices]),
                subset_name="train"
            )
            test_metrics = compute_evaluation_metrics(
                y_true=dataloader.y_test,
                y_pred=final_model.predict(dataloader.X_test[:, selected_indices]),
                subset_name="test"
            )

            run_stats = {
                "dataset": dataset_name,
                "run": n_runs,
                "method": "ga",
                "fs_time": round(fit_time, 2),
                "n_selected_features": len(selected_features),
                "best_cv_score": round(best_cv_score, 4),
                "ga_fitness_max": [history["fitness_max"]],
                "ga_fitness_avg": [history["fitness_avg"]],
                **train_metrics,
                **test_metrics
            }

            results = pd.concat([results, pd.DataFrame(run_stats)], ignore_index=True)
            save_results(results=results, output_file=output_file)

            if check_stopping_criteria(results=results, args=args, dataset_name=dataset_name, n_runs=n_runs):
                break

        del dataloader
        gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Genetic algorithm feature selection experiments."
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory path.")
    parser.add_argument("--metric-col", type=str, required=True, help="Metric to monitor for standard error threshold.")
    parser.add_argument("--se-thresh", type=float, default=0.03, help="Standard error threshold (default: 0.03).")
    parser.add_argument("--min-runs", type=int, default=5, help="Minimum number of runs per dataset (default: 5).")
    parser.add_argument("--max-runs", type=int, default=50, help="Maximum number of runs per dataset (default: 50).")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel workers for cross-validation (default: 1, -1 for all cores)."
    )
    parser.add_argument("--population-size", type=int, default=50, help="GA population size (default: 50).")
    parser.add_argument("--generations", type=int, default=40, help="Number of GA generations (default: 40).")
    parser.add_argument("--crossover-prob", type=float, default=0.8, help="Crossover probability (default: 0.8).")
    parser.add_argument("--mutation-prob", type=float, default=0.1, help="Mutation probability (default: 0.1).")
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Generations without improvement before early stopping (default: None, disabled)."
    )
    parser.add_argument("--is-debug", action="store_true", help="Debug mode: fewer datasets and 2 generations.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
