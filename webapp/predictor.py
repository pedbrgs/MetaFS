"""
Train one Ridge regression model per (target, FS method) on the full
metadataset and expose a predict() function that returns ranked algorithms
for all 5 criteria given a vector of 10 meta-features.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from meta_features import METAFEATURE_COLS

# ── Constants ────────────────────────────────────────────────────────────────

METHODS = [
    "ANOVA F-value",
    "Boruta",
    "CCEA",
    "Chi-Square",
    "Genetic Algorithm",
    "MRMR",
    "Mutual Information",
    "PCA",
]

TARGETS = [
    "MeanTestBalancedAccuracy",
    "MeanTestF1Score",
    "MeanCompressionRatio",
    "MeanFeatureSelectionTime",
]

TARGET_LABELS = {
    "MeanTestBalancedAccuracy": "Accuracy",
    "MeanTestF1Score": "F1 Score",
    "MeanCompressionRatio": "Compression",
    "MeanFeatureSelectionTime": "Time",
}

# True  → higher predicted value is better rank
TARGET_DIRECTION = {
    "MeanTestBalancedAccuracy": True,
    "MeanTestF1Score": True,
    "MeanCompressionRatio": True,
    "MeanFeatureSelectionTime": False,
}

LOG_TARGETS = {"MeanFeatureSelectionTime"}

_METADATASET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "results",
    "metadataset.parquet",
)
# If running from repo root (e.g. Render), results/ sits alongside app.py
if not os.path.exists(_METADATASET_PATH):
    _METADATASET_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results",
        "metadataset.parquet",
    )

# ── Model store ──────────────────────────────────────────────────────────────

# Populated by _train_models() at import time.
# Key: (target, method) → (StandardScaler, Ridge)
_models: dict[tuple[str, str], tuple[StandardScaler, Ridge]] = {}


def _train_models() -> None:
    df = pd.read_parquet(_METADATASET_PATH)

    for target in TARGETS:
        for method in METHODS:
            rows = df[df["MethodName"] == method].copy()
            rows = rows.dropna(subset=METAFEATURE_COLS + [target])

            if len(rows) < 2:
                continue

            X = rows[METAFEATURE_COLS].values.astype(float)
            y = rows[target].values.astype(float)

            if target in LOG_TARGETS:
                y = np.log1p(y)

            scaler = StandardScaler()
            X_sc = scaler.fit_transform(X)

            model = Ridge(alpha=1.0)
            model.fit(X_sc, y)

            _models[(target, method)] = (scaler, model)


# Train at import time (fast: 40 × 8 × 4 = 1280 rows, Ridge is instant).
_train_models()


# ── Public API ───────────────────────────────────────────────────────────────

def predict(
    meta_features: dict[str, float],
    accuracy_weight: float = 0.6,
    compression_weight: float = 0.4,
) -> dict:
    """
    Given the 10 meta-features of a new dataset, return the predicted ranking
    of FS algorithms for each of the 5 criteria.

    Parameters
    ----------
    meta_features : dict
        Maps each meta-feature name to its float value.
    accuracy_weight : float
        Weight for MeanTestBalancedAccuracy in the composite score.
    compression_weight : float
        Weight for MeanCompressionRatio in the composite score.

    Returns
    -------
    dict with keys:
      "rankings" → dict[criterion_label, list of {method, rank, score}]
      "meta_features" → dict[name, value]
    """
    x = np.array([meta_features[c] for c in METAFEATURE_COLS], dtype=float)

    # Predict raw scores per (target, method)
    raw_preds: dict[str, dict[str, float]] = {t: {} for t in TARGETS}
    for target in TARGETS:
        for method in METHODS:
            key = (target, method)
            if key not in _models:
                raw_preds[target][method] = np.nan
                continue
            scaler, model = _models[key]
            x_sc = scaler.transform(x.reshape(1, -1))
            pred = float(model.predict(x_sc)[0])
            if target in LOG_TARGETS:
                pred = float(np.expm1(pred))
            raw_preds[target][method] = pred

    # Build ranked lists for the 4 base criteria
    rankings: dict[str, list[dict]] = {}
    for target in TARGETS:
        label = TARGET_LABELS[target]
        scores = pd.Series(raw_preds[target])
        ascending = not TARGET_DIRECTION[target]
        ranks = scores.rank(ascending=ascending, method="min").astype(int)
        ranked = sorted(
            [
                {"method": m, "rank": int(ranks[m]), "score": round(scores[m], 6)}
                for m in METHODS
            ],
            key=lambda d: d["rank"],
        )
        rankings[label] = ranked

    # Composite criterion: weighted accuracy + compression
    total_w = accuracy_weight + compression_weight
    w_acc = accuracy_weight / total_w if total_w > 0 else 0.5
    w_cmp = compression_weight / total_w if total_w > 0 else 0.5

    acc_scores = pd.Series(raw_preds["MeanTestBalancedAccuracy"])
    cmp_scores = pd.Series(raw_preds["MeanCompressionRatio"])
    composite = w_acc * acc_scores + w_cmp * cmp_scores
    comp_ranks = composite.rank(ascending=False, method="min").astype(int)
    rankings["Composite"] = sorted(
        [
            {
                "method": m,
                "rank": int(comp_ranks[m]),
                "score": round(float(composite[m]), 6),
            }
            for m in METHODS
        ],
        key=lambda d: d["rank"],
    )

    return {"rankings": rankings}
