# CRISP-DM Report — BRFSS 2015 Diabetes

**Author:** Michael Kennedy • **Run:** 2025-09-14T04:08:51.686536Z

## 1. Business Understanding
- Goal: Predict diabetes indicator from BRFSS features to support screening outreach.
- KPIs: Accuracy, **macro-F1** (class balance), **calibration** (Brier/curves).
- Constraints: 8 GB RAM; Colab-compatible; deterministic; MIT-licensed.

## 2. Data Understanding
- Source: BRFSS 2015 health indicators (course dataset).
- Target: `Diabetes_binary`.
- Class distribution: see `reports/figures/class_distribution.png`.
- Top-variance correlation snapshot: `reports/figures/corr_heatmap_top20.png`.
- Univariate top features: `reports/figures/univariate_top6.png`.
- Simple importance proxy: `reports/figures/feature_importance_proxy.png`.

## 3. Data Preparation
- Deterministic **70/15/15** split (stratified).
- Standardization via `StandardScaler`.
- **Class weights** computed from training fold to mitigate imbalance.

## 4. Modeling
### Scikit baselines
- Logistic Regression (`class_weight=balanced`)
- HistGradientBoosting

### Deep Learning (PyTorch MLP)
- 2×hidden layers (256 units), ReLU, Dropout 0.25
- AdamW, cosine LR, **AMP**, **early stopping** on val macro-F1

## 5. Evaluation
- **Baselines**: see `reports/baseline_logreg_metrics.json` and `reports/baseline_hgb_metrics.json`.
- **DL (MLP)**: `reports/mlp_metrics.json`.
- Confusion matrices:
  - Logistic Regression: `reports/figures/cm_logreg_test.png`
  - MLP: `reports/figures/cm_mlp_test.png`
- If binary:
  - ROC/PR/Calibration (LogReg): `reports/figures/roc_logreg_test.png`, `reports/figures/pr_logreg_test.png`, `reports/figures/calibration_logreg_test.png`
  - ROC/PR/Calibration (MLP): `reports/figures/roc_mlp_test.png`, `reports/figures/pr_mlp_test.png`, `reports/figures/calibration_mlp_test.png`



## 5.4 Results Comparison Table

| Model | Accuracy | Macro-F1 | AUC-ROC | Avg Precision | Brier |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | - | - | - | - | - |
| HistGradientBoosting | - | - | - | - | - |
| PyTorch MLP | - | - | - | - | - |

## 6. Deployment / Reproducibility
- All figures saved under `reports/figures/`.
- Metrics exported as JSON for grading and comparison.
- Colab badge in README; single `Makefile` entry points: `baselines`, `train`, `eval`.

## Limitations & Ethics
- Population/sample bias; self-reported BRFSS signals; risk of overfitting to survey artifacts.
- This work is **decision-support only**, **not medical advice**.


## Ethical Considerations

- BRFSS is self-reported survey data → recall bias, demographic sampling issues.

- Privacy: dataset is anonymized but merging with other data could risk re-identification.

- Fairness: need to check subgroup performance.

- Limitations: decision-support only, **not medical advice**.


## Citation

Alex Teboul. *“Diabetes Health Indicators Dataset (BRFSS 2015 Survey)”*. Kaggle, 2021.  
[https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset).  
Accessed: 2025-09-14.

