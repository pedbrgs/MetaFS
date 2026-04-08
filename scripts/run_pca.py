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
from typing import Callable
from sklearn.base import ClassifierMixin
from sklearn.base import clone as sklearn_clone
from sklearn.decomposition import PCA
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
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

from pyccea.utils.datasets import DataLoader

MAX_FS_HOURS = 12
MAX_FS_SECONDS = MAX_FS_HOURS * 3600


class _FoldData:
    """Picklable proxy carrying only what PCAReducer.tune() needs."""
    def __init__(self, folds, n_features, kfolds):
        self._folds = folds
        self.n_features = n_features
        self.kfolds = kfolds

    def get_fold(self, fold_idx, normalize=False):
        return self._folds[fold_idx]


def _pca_worker(result_queue, fold_data, X_train, y_train, n_features, random_state, n_jobs, step_size):
    """Run PCA tuning and reduction in an isolated process.

    Calls os.setpgrp() so the entire process group (including joblib workers)
    can be killed atomically with os.killpg() on timeout.
    """
    os.setpgrp()
    try:
        base_model = RandomForestClassifier(random_state=random_state, class_weight="balanced", n_jobs=n_jobs)
        reducer = PCAReducer(
            estimator=base_model,
            random_state=random_state,
        )
        t0 = time.time()
        tuning_results = reducer.tune(
            dataloader=fold_data,
            eval_function=balanced_accuracy_score,
            step_size=step_size,
        )
        tuning_time = time.time() - t0

        t1 = time.time()
        X_train_reduced = reducer.transform(
            X_train=X_train,
            n_components=tuning_results["best_k"],
        )
        fs_time = time.time() - t1

        result_queue.put(("ok", tuning_results, X_train_reduced, tuning_time, fs_time))
    except Exception as e:
        result_queue.put(("error", str(e)))


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


class PCAReducer:
    """PCA-based dimensionality reduction with cross-validated component tuning.

    Attributes
    ----------
    estimator : ClassifierMixin
        An unfitted classifier with `predict` and `predict_proba` methods.
    random_state : int, optional
        Random seed used to initialize the estimator and PCA, by default 1234.
    """

    def __init__(
        self,
        estimator: ClassifierMixin,
        random_state: int = 1234,
    ):
        """Init PCA reducer."""
        self.base_estimator = estimator
        self.random_state = random_state

    def transform(
        self,
        X_train: np.ndarray,
        n_components: int,
        X_test: np.ndarray = None,
    ):
        """Fit PCA on X_train and return the reduced representation(s).

        A StandardScaler is applied internally before PCA.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix.
        n_components : int
            Number of principal components to keep.
        X_test : np.ndarray, optional
            Test feature matrix to transform with the same scaler/PCA.

        Returns
        -------
        X_train_reduced : np.ndarray
            Reduced training matrix.
        X_test_reduced : np.ndarray or None
            Reduced test matrix (only when X_test is provided).
        pca : PCA
            Fitted PCA object (exposes explained_variance_ratio_ etc.).
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        pca = PCA(n_components=n_components, random_state=self.random_state)
        X_train_reduced = pca.fit_transform(X_train_scaled)

        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            X_test_reduced = pca.transform(X_test_scaled)
            return X_train_reduced, X_test_reduced, pca

        return X_train_reduced, None, pca

    def _reset_estimator(self) -> ClassifierMixin:
        """Return a fresh, unfitted copy of the base estimator."""
        return sklearn_clone(self.base_estimator)

    def tune(
        self,
        dataloader,
        eval_function: Callable,
        step_size: float = 0.05,
    ) -> dict:
        """Tune the number of PCA components using k-fold cross-validation.

        Parameters
        ----------
        dataloader : DataLoader
            Prepared DataLoader instance with k-fold splits available.
        eval_function : Callable
            Scoring function with signature (y_true, y_pred) -> float.
        step_size : float, optional
            Fractional step between component counts to evaluate, by default 0.05.

        Returns
        -------
        dict
            best_k, k_values, cv_scores, cv_stds.
        """
        k_values = []
        cv_scores = []
        cv_stds = []

        n_features = dataloader.n_features

        n_steps = int(1.0 / step_size)
        k_percentages = np.linspace(step_size, 1.0, n_steps)[:-1]
        k_components = [max(1, int(p * n_features)) for p in k_percentages]

        for k in k_components:
            logging.info(f"Evaluating k={k} components ({k / n_features * 100:.1f}%)")
            fold_scores = []

            for fold_idx in range(dataloader.kfolds):
                X_f_train, y_f_train, X_f_val, y_f_val = dataloader.get_fold(fold_idx, normalize=False)

                X_f_train_red, X_f_val_red, _ = self.transform(
                    X_train=X_f_train,
                    n_components=k,
                    X_test=X_f_val,
                )

                estimator = self._reset_estimator()
                estimator.fit(X_f_train_red, y_f_train)

                y_pred = estimator.predict(X_f_val_red)
                score = eval_function(y_true=y_f_val, y_pred=y_pred)
                fold_scores.append(score)

            cv_mean = np.mean(fold_scores)
            cv_std = np.std(fold_scores)

            k_values.append(k)
            cv_scores.append(cv_mean)
            cv_stds.append(cv_std)

        best_k_index = np.argmax(cv_scores)
        return {
            "best_k": k_values[best_k_index],
            "k_values": k_values,
            "cv_scores": cv_scores,
            "cv_stds": cv_stds
        }


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
    errors = results.groupby("dataset")[metric_col].apply(cumulative_standard_error)
    achieved_errors = errors.groupby(level=0).last()
    run_counts = results.groupby("dataset").size()
    cond_error_met = (achieved_errors <= standard_error_threshold) & (run_counts >= min_runs)
    cond_error_failed = (achieved_errors > standard_error_threshold) & (run_counts >= max_runs)
    completed_mask = cond_error_met | cond_error_failed
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

    output_file = "pca_experiments.parquet"

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

            # Initialize PCA reducer
            base_model = RandomForestClassifier(random_state=random_state, class_weight="balanced", n_jobs=args.n_jobs)
            reducer = PCAReducer(
                estimator=base_model,
                random_state=random_state,
            )

            # Pre-extract folds as numpy arrays so they can be pickled into the subprocess.
            folds = [dataloader.get_fold(i, normalize=False) for i in range(dataloader.kfolds)]
            fold_data = _FoldData(
                folds=folds,
                n_features=dataloader.n_features,
                kfolds=dataloader.kfolds,
            )

            result_queue = mp.Queue()
            proc = mp.Process(
                target=_pca_worker,
                args=(
                    result_queue, fold_data,
                    dataloader.X_train, dataloader.y_train,
                    dataloader.n_features, random_state, args.n_jobs,
                    0.25 if args.is_debug else 0.05,
                ),
            )
            start_tuning = time.time()
            proc.start()
            proc.join(timeout=MAX_FS_SECONDS)

            if proc.is_alive():
                elapsed = round(time.time() - start_tuning, 2)
                try:
                    os.killpg(proc.pid, 9)  # SIGKILL to entire process group
                except ProcessLookupError:
                    pass
                proc.join()
                logging.warning(f"Run #{n_runs} exceeded {MAX_FS_HOURS}h time limit.")
                run_stats = {
                    "dataset": dataset_name,
                    "run": n_runs,
                    "method": "pca",
                    "tuning_time": None,
                    "fs_time": elapsed,
                    "n_components": None,
                    "feature_values": None,
                    "cv_avgs": None,
                    "cv_stds": None,
                }
                results = pd.concat([results, pd.DataFrame([run_stats])], ignore_index=True)
                save_results(results=results, output_file=output_file)
                break

            outcome = result_queue.get() if not result_queue.empty() else ("error", "Worker produced no result")
            if outcome[0] == "error":
                logging.error(f"PCA reduction failed: {outcome[1]}")
                break
            _, tuning_results, X_train_reduced, tuning_time, fs_time = outcome
            logging.info(f"Best k={tuning_results['best_k']} components selected by tuning.")

            # Transform test set with the best number of components
            _, X_test_reduced, pca = reducer.transform(
                X_train=dataloader.X_train,
                n_components=tuning_results["best_k"],
                X_test=dataloader.X_test,
            )

            # Train final model on reduced features
            final_model = reducer._reset_estimator()
            final_model.fit(X_train_reduced, dataloader.y_train)

            # Evaluate
            train_metrics = compute_evaluation_metrics(
                y_true=dataloader.y_train,
                y_pred=final_model.predict(X_train_reduced),
                subset_name="train"
            )
            test_metrics = compute_evaluation_metrics(
                y_true=dataloader.y_test,
                y_pred=final_model.predict(X_test_reduced),
                subset_name="test"
            )

            run_stats = {
                "dataset": dataset_name,
                "run": n_runs,
                "method": "pca",
                "tuning_time": round(tuning_time, 2),
                "fs_time": round(fs_time, 2),
                "n_components": tuning_results["best_k"],
                "explained_variance_ratio": [pca.explained_variance_ratio_.tolist()],
                "cumulative_explained_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
                "feature_values": [tuning_results["k_values"]],
                "cv_avgs": [tuning_results["cv_scores"]],
                "cv_stds": [tuning_results["cv_stds"]],
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
        description="PCA-based dimensionality reduction experiments."
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
        help="Number of parallel workers for the RandomForest classifier (default: 1, -1 for all cores)."
    )
    parser.add_argument("--is-debug", action="store_true", help="Debug mode: fewer datasets and steps.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
