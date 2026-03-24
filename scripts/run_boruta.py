import gc
import os
import time
import random
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from typing import List
from boruta import BorutaPy
from sklearn.base import ClassifierMixin
from sklearn.base import clone as sklearn_clone
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

warnings.filterwarnings(action="ignore", category=DeprecationWarning)
warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

from pyccea.utils.datasets import DataLoader


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


class BorutaFeatureSelector:
    """Boruta wrapper feature selection using a Random Forest estimator.

    Attributes
    ----------
    estimator : ClassifierMixin
        An unfitted classifier compatible with BorutaPy (must expose feature importances).
    perc : int, optional
        Percentile of the shadow feature importance distribution used as the
        acceptance threshold, by default 100.
    alpha : float, optional
        Family-wise error rate (FWER) significance level, by default 0.05.
    max_iter : int, optional
        Maximum number of Boruta iterations, by default 1000.
    early_stopping : bool, optional
        Whether to stop early if feature selection stabilises, by default True.
    n_iter_no_change : int, optional
        Number of iterations without change required to trigger early stopping,
        by default 100.
    random_state : int, optional
        Random seed, by default 1234.
    """

    def __init__(
        self,
        estimator: ClassifierMixin,
        perc: int = 100,
        alpha: float = 0.05,
        max_iter: int = 1000,
        early_stopping: bool = True,
        n_iter_no_change: int = 100,
        random_state: int = 1234,
    ):
        """Init Boruta feature selector."""
        self.base_estimator = estimator
        self.perc = perc
        self.alpha = alpha
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state

    def _reset_estimator(self) -> ClassifierMixin:
        """Return a fresh, unfitted copy of the base estimator."""
        return sklearn_clone(self.base_estimator)

    def select(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: List[str],
    ) -> List[str]:
        """Run Boruta and return selected feature names.

        Returns confirmed features. If none are confirmed, falls back to
        tentative features. If still empty, keeps all features.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Target labels.
        feature_names : List[str]
            Names for each feature column.

        Returns
        -------
        List[str]
            Selected feature names.
        """
        estimator = self._reset_estimator()
        boruta = BorutaPy(
            estimator=estimator,
            n_estimators="auto",
            perc=self.perc,
            alpha=self.alpha,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state
        )
        boruta.fit(X_train, y_train)

        selected_mask = boruta.support_

        if not selected_mask.any():
            logging.warning("Boruta confirmed no features. Falling back to tentative features.")
            selected_mask = boruta.support_weak_
        if not selected_mask.any():
            logging.warning("Boruta found no tentative features either. Keeping all features.")
            selected_mask = np.ones(len(feature_names), dtype=bool)

        return [name for name, sel in zip(feature_names, selected_mask) if sel]


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

    output_file = "boruta_experiments.parquet"

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

            # Initialize Boruta selector
            base_model = RandomForestClassifier(
                random_state=random_state,
                class_weight="balanced"
            )
            selector = BorutaFeatureSelector(
                estimator=base_model,
                perc=args.perc,
                alpha=args.alpha,
                max_iter=args.max_iter,
                early_stopping=args.early_stopping,
                n_iter_no_change=args.n_iter_no_change,
                random_state=random_state,
            )

            # Run Boruta feature selection on full training set
            start_fs = time.time()
            selected_features = selector.select(
                X_train=dataloader.X_train,
                y_train=dataloader.y_train,
                feature_names=feature_names,
            )
            fs_time = time.time() - start_fs
            logging.info(f"Boruta selected {len(selected_features)} features.")

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
                "method": "boruta",
                "fs_time": round(fs_time, 2),
                "n_selected_features": len(selected_features),
                **train_metrics,
                **test_metrics
            }

            results = pd.concat([results, pd.DataFrame([run_stats])], ignore_index=True)
            save_results(results=results, output_file=output_file)

            if check_stopping_criteria(results=results, args=args, dataset_name=dataset_name, n_runs=n_runs):
                break

        del dataloader
        gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Boruta wrapper feature selection experiments."
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory path.")
    parser.add_argument("--metric-col", type=str, required=True, help="Metric to monitor for standard error threshold.")
    parser.add_argument("--se-thresh", type=float, default=0.03, help="Standard error threshold (default: 0.03).")
    parser.add_argument("--min-runs", type=int, default=5, help="Minimum number of runs per dataset (default: 5).")
    parser.add_argument("--max-runs", type=int, default=50, help="Maximum number of runs per dataset (default: 50).")
    parser.add_argument("--perc", type=int, default=100, help="Percentile of shadow importance used as threshold (default: 100).")
    parser.add_argument("--alpha", type=float, default=0.05, help="FWER significance level for Boruta (default: 0.05).")
    parser.add_argument("--max-iter", type=int, default=1000, help="Maximum number of Boruta iterations (default: 1000).")
    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=True, help="Enable early stopping in Boruta (default: True).")
    parser.add_argument("--n-iter-no-change", type=int, default=100, help="Iterations without change to trigger early stopping (default: 100).")
    parser.add_argument("--is-debug", action="store_true", help="Debug mode: fewer datasets.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
