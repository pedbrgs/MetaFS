import gc
import os
import time
import random
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from functools import partial
from typing import Callable, List
from sklearn.base import ClassifierMixin
from sklearn.base import clone as sklearn_clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

from pyccea.utils.datasets import DataLoader

VALID_METHODS = ("mi", "chi2", "f")


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


class FilterFeatureSelector:
    """Sklearn filter-based feature selection (Mutual Information, Chi-squared, F-statistic).

    Attributes
    ----------
    estimator : ClassifierMixin
        An unfitted classifier with `predict` and `predict_proba` methods.
    method : str
        Feature selection method: 'mi' (Mutual Information), 'chi2' (Chi-squared),
        or 'f' (F-statistic).
    random_state : int, optional
        Random seed used to initialize the estimator and/or selector, by default 1234.
    n_jobs : int, optional
        Number of parallel workers. Only used for method='mi'. By default 1.
    """

    def __init__(
        self,
        estimator: ClassifierMixin,
        method: str,
        random_state: int = 1234,
        n_jobs: int = 1
    ):
        """Init filter feature selector."""
        if method not in VALID_METHODS:
            raise ValueError(f"method must be one of {VALID_METHODS}, got '{method}'.")
        self.base_estimator = estimator
        self.method = method
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _build_score_func(self) -> Callable:
        """Return the sklearn score function for the chosen method."""
        if self.method == "mi":
            return partial(
                mutual_info_classif,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        elif self.method == "chi2":
            return chi2
        elif self.method == "f":
            return f_classif

    def select(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
        n_features: int
    ) -> List[str]:
        """Select the top `n_features` most relevant features.

        For chi2, a MinMaxScaler is applied internally to ensure non-negative values.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Target labels.
        feature_names : List[str]
            Names for each feature column.
        n_features : int
            Number of top features to select.

        Returns
        -------
        List[str]
            Selected feature names.
        """
        X = X_train.copy()

        if self.method == "chi2":
            # chi2 requires non-negative features; apply MinMaxScaler
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)

        score_func = self._build_score_func()
        selector = SelectKBest(score_func=score_func, k=n_features)
        selector.fit(X, y_train)

        selected_mask = selector.get_support()
        return [name for name, selected in zip(feature_names, selected_mask) if selected]

    def _reset_estimator(self) -> ClassifierMixin:
        """Return a fresh, unfitted copy of the base estimator."""
        return sklearn_clone(self.base_estimator)

    def tune(
        self,
        dataloader,
        eval_function: Callable,
        step_size: float = 0.05,
    ) -> dict:
        """Tune the number of features using k-fold cross-validation.

        Parameters
        ----------
        dataloader : DataLoader
            Prepared DataLoader instance with k-fold splits available.
        eval_function : Callable
            Scoring function with signature (y_true, y_pred) -> float.
        step_size : float, optional
            Fractional step between k values to evaluate, by default 0.05.

        Returns
        -------
        dict
            best_k, k_values, cv_scores, cv_stds.
        """
        if self.method != "mi" and self.n_jobs != 1:
            logging.warning(
                f"n_jobs={self.n_jobs} is only supported for method='mi'. "
                f"Ignoring n_jobs for method='{self.method}'."
            )

        k_values = []
        cv_scores = []
        cv_stds = []

        n_features = dataloader.n_features
        feature_names = [f"feat_{i}" for i in range(n_features)]

        n_steps = int(1.0 / step_size)
        k_percentages = np.linspace(step_size, 1.0, n_steps)[:-1]
        k_features = [max(1, int(p * n_features)) for p in k_percentages]

        for k in k_features:
            logging.info(f"Evaluating k={k} features ({k / n_features * 100:.1f}%)")
            fold_scores = []

            for fold_idx in range(dataloader.kfolds):
                X_f_train, y_f_train, X_f_val, y_f_val = dataloader.get_fold(fold_idx, normalize=False)

                selected_features = self.select(
                    X_train=X_f_train,
                    y_train=y_f_train,
                    feature_names=feature_names,
                    n_features=k
                )

                # Get column indices for selected features
                selected_indices = [feature_names.index(f) for f in selected_features]

                estimator = self._reset_estimator()
                estimator.fit(X_f_train[:, selected_indices], y_f_train)

                y_pred = estimator.predict(X_f_val[:, selected_indices])
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
    metric_series = results[results["dataset"] == dataset_name][args.metric_col]
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

    output_file = f"filter_experiments_{args.method}.parquet"

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

            # Initialize filter selector
            base_model = RandomForestClassifier(random_state=random_state, class_weight="balanced")
            selector = FilterFeatureSelector(
                estimator=base_model,
                method=args.method,
                random_state=random_state,
                n_jobs=args.n_jobs
            )

            # Tune number of features
            start_tuning = time.time()
            tuning_results = selector.tune(
                dataloader=dataloader,
                eval_function=balanced_accuracy_score,
                step_size=0.25 if args.is_debug else 0.05
            )
            tuning_time = time.time() - start_tuning
            best_k = tuning_results["best_k"]
            logging.info(f"Best k={best_k} features selected by tuning.")

            # Select best k features on full training set
            start_fs = time.time()
            selected_features = selector.select(
                X_train=dataloader.X_train,
                y_train=dataloader.y_train,
                feature_names=feature_names,
                n_features=best_k
            )
            fs_time = time.time() - start_fs

            selected_indices = [feature_names.index(f) for f in selected_features]

            # Train final model on selected features
            final_model = selector._reset_estimator()
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
                "method": args.method,
                "tuning_time": round(tuning_time, 2),
                "fs_time": round(fs_time, 2),
                "n_selected_features": best_k,
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
        description="Filter-based feature selection experiments (MI, Chi-squared, F-statistic)."
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory path.")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=list(VALID_METHODS),
        help="Feature selection method: 'mi' (Mutual Information), 'chi2' (Chi-squared), 'f' (F-statistic)."
    )
    parser.add_argument("--metric-col", type=str, required=True, help="Metric to monitor for standard error threshold.")
    parser.add_argument("--se-thresh", type=float, default=0.03, help="Standard error threshold (default: 0.03).")
    parser.add_argument("--min-runs", type=int, default=5, help="Minimum number of runs per dataset (default: 5).")
    parser.add_argument("--max-runs", type=int, default=50, help="Maximum number of runs per dataset (default: 50).")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel workers. Only used for method='mi' (default: 1, -1 for all cores)."
    )
    parser.add_argument("--is-debug", action="store_true", help="Debug mode: fewer datasets and steps.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
