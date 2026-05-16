
<div align="center">

# MetaFS

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-orange)](https://huggingface.co/spaces/pedbrgs/metafs)

</div>

---

## 💡 Overview

Choosing the right feature selection algorithm for a given dataset is a non-trivial task that typically requires extensive experimentation. **MetaFS** addresses this by framing algorithm selection as a meta-learning problem: given a new dataset, a set of meta-features is extracted and used to predict the performance of eight feature selection algorithms across five evaluation criteria. MetaFS is available as a [web application](https://huggingface.co/spaces/pedbrgs/metafs) on Hugging Face Spaces. Upload your dataset, select the target column, adjust the composite criterion weights, and get algorithm rankings in seconds.

---

## 🧠 Methodology

Meta-learners are Ridge regression models trained on a metadataset of [40 benchmark classification datasets](https://github.com/pedbrgs/High-Dimensional-Datasets) under a Leave-One-Dataset-Out (LODO) evaluation protocol. Predictions are generated instantly at inference time, providing ranked recommendations without running any feature selection method on the target dataset.

---

## 🔬 Algorithms

The following feature selection methods are evaluated and ranked by MetaFS:

| Algorithm | Package | Reference |
|---|---|---|
| ANOVA F-value | scikit-learn | `SelectKBest(f_classif)` |
| Boruta | boruta | `BorutaPy` |
| CCEA | pyccea | `CCEA` |
| Chi-Square | scikit-learn | `SelectKBest(chi2)` |
| Genetic Algorithm | deap | `algorithms.eaSimple` |
| MRMR | mrmr-selection | `mrmr_classif` |
| Mutual Information | scikit-learn | `SelectKBest(mutual_info_classif)` |
| PCA | scikit-learn | `PCA` |

---

## 📊 Evaluation Criteria

| Criterion | Direction | Description |
|---|---|---|
| Balanced Accuracy | ↑ Higher is better | Average per-class accuracy after feature selection, robust to class imbalance. |
| F1 Score | ↑ Higher is better | Harmonic mean of precision and recall after feature selection. |
| Compression Ratio | ↑ Higher is better | Fraction of features removed; 1 means all features discarded. |
| Feature Selection Time | ↓ Lower is better | Wall-clock time (seconds) to run the feature selection algorithm. |
| Composite | ↑ Higher is better | User-weighted average of Balanced Accuracy and Compression Ratio. |

---

## ⚙️ Meta-features

MetaFS extracts the following 10 dataset meta-features to characterize each new dataset:

| Meta-feature | Description |
|---|---|
| `ImbalanceRatio` | Ratio between the largest and smallest class counts. |
| `MaxFeatureClassSpearman` | Maximum absolute Spearman correlation between any feature and the target. |
| `MF_Dimensionality` | Ratio of features to samples. |
| `MF_MaxNumericMutualInformation` | Maximum mutual information between any feature and the target. |
| `MF_MaxCardinalityOfNumericFeatures` | Maximum number of unique values across all numeric features. |
| `MF_StdevNumericMutualInformation` | Standard deviation of mutual information scores across features. |
| `MF_Quartile1ClassProbability` | First quartile of the class probability distribution. |
| `MF_MinClassProbability` | Minimum class probability. |
| `MF_MaxNumericJointEntropy` | Maximum joint entropy between any feature and the target. |
| `MF_KurtosisClassProbability` | Kurtosis of the class probability distribution (Fisher's excess). |

---

## 📜 Citation

If you use MetaFS in your research, please cite:

```bibtex
@misc{MetaFS,
    author = {Venâncio, Pedro},
    title = {{MetaFS}: A meta-learning approach to feature selection algorithm recommendation for binary classification tasks},
    year = {2026},
    publisher = {GitHub},
    url = {https://github.com/pedbrgs/MetaFS}
}
```

---

## 📫 Contact

Please send any bug reports, questions or suggestions directly in the repository.
