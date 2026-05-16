"""
Compute the 10 meta-features used by the Ridge meta-learners.

Implementations match the metalearn library (byu-dml/metalearn):
  - Binning: round(n^(1/3)) equal-width bins  (pd.cut equivalent)
  - MI: mutual_info_score on binned features   (sklearn.metrics)
  - Joint entropy: scipy.stats.entropy on (bin, class) pair counts
  - Kurtosis: scipy.stats.kurtosis (Fisher excess, default)

Optimisations over the original loop-based approach:
  - Spearman: vectorised via numpy rank + single matrix multiply
  - MI / JE: numpy digitize + bincount  (no pandas per-feature overhead)
  - Row sampling: datasets > MAX_SAMPLES are stratified-sampled before
    computing any feature, keeping the distribution intact
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as scipy_kurtosis
from sklearn.preprocessing import LabelEncoder

MAX_SAMPLES = 5_000  # rows cap for meta-feature computation

METAFEATURE_COLS = [
    "ImbalanceRatio",
    "MaxFeatureClassSpearman",
    "MF_Dimensionality",
    "MF_MaxNumericMutualInformation",
    "MF_MaxCardinalityOfNumericFeatures",
    "MF_StdevNumericMutualInformation",
    "MF_Quartile1ClassProbability",
    "MF_MinClassProbability",
    "MF_MaxNumericJointEntropy",
    "MF_KurtosisClassProbability",
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _spearman_max_with_target(X: np.ndarray, y: np.ndarray) -> float:
    """
    Max |Spearman(col, y)| over all columns of X, fully vectorised.

    Rank each column of X and y once, then compute all correlations
    with a single matrix multiply – O(n·p) instead of O(n·p·log n) calls.
    NaN columns are skipped.
    """
    from scipy.stats import rankdata

    n, p = X.shape
    if p == 0 or n < 3:
        return 0.0

    y_r = rankdata(y).astype(np.float64)
    y_r -= y_r.mean()
    y_norm = np.sqrt((y_r ** 2).sum())
    if y_norm == 0:
        return 0.0

    # Rank all columns at once; columns may have NaN → handle per-column
    X_r = np.apply_along_axis(rankdata, 0, X).astype(np.float64)
    X_r -= X_r.mean(axis=0)
    X_norms = np.sqrt((X_r ** 2).sum(axis=0))
    X_norms[X_norms == 0] = np.inf  # avoid div-by-zero; result → 0 corr

    corrs = (X_r * y_r[:, np.newaxis]).sum(axis=0) / (X_norms * y_norm)
    return float(np.nanmax(np.abs(corrs)))


def _mi_and_je_all(X: np.ndarray, y_enc: np.ndarray) -> tuple[list, list]:
    """
    Compute MI and joint-entropy for every column of X against y_enc
    using numpy operations only (no pandas per-feature loop overhead).

    Binning: round(n^(1/3)) equal-width intervals, matching pd.cut.
    MI formula: sum p(x,y) * ln(p(x,y)/(p(x)*p(y)))  [nats, like sklearn]
    JE formula: -sum p(x,y) * ln(p(x,y))              [nats, like scipy]
    """
    n, p = X.shape
    n_classes = int(y_enc.max()) + 1
    n_bins = max(2, round(n ** (1.0 / 3.0)))

    mi_list: list[float] = []
    je_list: list[float] = []

    for col_idx in range(p):
        col = X[:, col_idx]
        valid = ~np.isnan(col)
        c = col[valid]
        yc = y_enc[valid]
        nv = len(c)

        if nv < 2:
            continue

        col_min, col_max = c.min(), c.max()
        if col_min == col_max:
            continue

        # Equal-width bin edges (equivalent to pd.cut with right=True)
        edges = np.linspace(col_min, col_max, n_bins + 1)
        edges[0] -= 1e-10  # include leftmost point

        bins = np.clip(np.digitize(c, edges) - 1, 0, n_bins - 1)

        # 2-D contingency via bincount: shape (n_bins, n_classes)
        joint_counts = np.bincount(
            bins * n_classes + yc, minlength=n_bins * n_classes
        ).reshape(n_bins, n_classes)

        # ── Mutual information ────────────────────────────────────────────
        p_xy = joint_counts / nv  # (n_bins, n_classes)
        p_x = p_xy.sum(axis=1, keepdims=True)
        p_y = p_xy.sum(axis=0, keepdims=True)
        denom = p_x * p_y
        mask = (p_xy > 0) & (denom > 0)
        mi = float(np.sum(p_xy[mask] * np.log(p_xy[mask] / denom[mask])))
        mi_list.append(mi)

        # ── Joint entropy ─────────────────────────────────────────────────
        p_flat = p_xy[p_xy > 0]
        je = float(-np.sum(p_flat * np.log(p_flat)))
        je_list.append(je)

    return mi_list, je_list


def _stratified_sample(df: pd.DataFrame, target_col: str, n: int, seed: int = 42) -> pd.DataFrame:
    """Return up to `n` rows, stratified by target_col."""
    if len(df) <= n:
        return df
    rng = np.random.default_rng(seed)
    frames = []
    for _, grp in df.groupby(target_col, sort=False):
        k = max(1, round(n * len(grp) / len(df)))
        idx = rng.choice(len(grp), size=min(k, len(grp)), replace=False)
        frames.append(grp.iloc[idx])
    return pd.concat(frames).sample(frac=1, random_state=seed)  # shuffle


# ── Public API ────────────────────────────────────────────────────────────────

def compute_meta_features(df: pd.DataFrame, target_col: str, use_sampling: bool = True) -> dict:
    """
    Compute the 10 meta-features from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including the target column.
    target_col : str
        Name of the target (class label) column.

    Returns
    -------
    dict mapping each meta-feature name to its float value.
    """
    # ── Sampling (only when enabled and dataset is large) ────────────────
    if use_sampling and len(df) > MAX_SAMPLES:
        df = _stratified_sample(df, target_col, MAX_SAMPLES)

    Y = df[target_col].astype(str)
    numeric_cols = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).columns.tolist()
    X_num = df[numeric_cols]

    n_samples = len(df)
    n_features = len(numeric_cols)

    # ── 1. ImbalanceRatio ────────────────────────────────────────────────
    class_counts = Y.value_counts()
    imbalance_ratio = (
        float(class_counts.max()) / float(class_counts.min())
        if len(class_counts) > 1 else 1.0
    )

    # ── Class probability distribution (shared by 7, 8, 10) ─────────────
    probs = (class_counts / n_samples).values.astype(np.float64)

    # ── 7. MF_Quartile1ClassProbability ─────────────────────────────────
    mf_q1_class_prob = float(np.percentile(probs, 25))

    # ── 8. MF_MinClassProbability ────────────────────────────────────────
    mf_min_class_prob = float(probs.min())

    # ── 10. MF_KurtosisClassProbability ─────────────────────────────────
    mf_kurtosis_class_prob = float(scipy_kurtosis(probs)) if len(probs) > 1 else 0.0

    # ── 3. MF_Dimensionality ─────────────────────────────────────────────
    mf_dimensionality = n_features / n_samples if n_samples > 0 else 0.0

    # ── 5. MF_MaxCardinalityOfNumericFeatures ────────────────────────────
    if n_features > 0:
        mf_max_cardinality = float(X_num.nunique().max())
    else:
        mf_max_cardinality = 0.0

    if n_features == 0:
        return {
            "ImbalanceRatio": imbalance_ratio,
            "MaxFeatureClassSpearman": 0.0,
            "MF_Dimensionality": mf_dimensionality,
            "MF_MaxNumericMutualInformation": 0.0,
            "MF_MaxCardinalityOfNumericFeatures": mf_max_cardinality,
            "MF_StdevNumericMutualInformation": 0.0,
            "MF_Quartile1ClassProbability": mf_q1_class_prob,
            "MF_MinClassProbability": mf_min_class_prob,
            "MF_MaxNumericJointEntropy": 0.0,
            "MF_KurtosisClassProbability": mf_kurtosis_class_prob,
        }

    X_arr = X_num.values.astype(np.float64)
    y_codes = pd.factorize(Y)[0].astype(np.float64)

    # ── 2. MaxFeatureClassSpearman (vectorised) ──────────────────────────
    max_feature_class_spearman = _spearman_max_with_target(X_arr, y_codes)

    # ── 4, 6, 9. MI and JE via numpy (no pandas loop overhead) ───────────
    y_enc = LabelEncoder().fit_transform(Y.values)
    mi_scores, joint_entropies = _mi_and_je_all(X_arr, y_enc)

    mf_max_mi = max(mi_scores) if mi_scores else 0.0
    mf_stdev_mi = float(np.std(mi_scores, ddof=1)) if len(mi_scores) > 1 else 0.0
    mf_max_je = max(joint_entropies) if joint_entropies else 0.0

    return {
        "ImbalanceRatio": imbalance_ratio,
        "MaxFeatureClassSpearman": max_feature_class_spearman,
        "MF_Dimensionality": mf_dimensionality,
        "MF_MaxNumericMutualInformation": mf_max_mi,
        "MF_MaxCardinalityOfNumericFeatures": mf_max_cardinality,
        "MF_StdevNumericMutualInformation": mf_stdev_mi,
        "MF_Quartile1ClassProbability": mf_q1_class_prob,
        "MF_MinClassProbability": mf_min_class_prob,
        "MF_MaxNumericJointEntropy": mf_max_je,
        "MF_KurtosisClassProbability": mf_kurtosis_class_prob,
    }
