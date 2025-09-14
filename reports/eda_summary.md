# EDA Summary — BRFSS 2015 Diabetes (Binary)

**Timestamp:** 2025-09-14T04:00:03.726564Z

## Dataset
- File: `diabetes_binary_health_indicators_BRFSS2015.csv`
- Shape: 253680 rows × 22 columns
- Missing values (total): 0
- Duplicated rows: 24206
- Target column: `Diabetes_binary`
- Class distribution: {
  "0.0": 218334,
  "1.0": 35346
}

## Notes
- All features appear numeric; scaling not strictly required for tree models but recommended for linear/NN models.
- Consider class imbalance mitigation if needed (e.g., class weights).

## Figures
- `reports/figures/class_distribution.png`
- `reports/figures/corr_heatmap_top20.png`

## Additional EDA Artifacts
- reports/figures/univariate_top6.png
- reports/figures/feature_importance_proxy.png
- reports/figures/permutation_importance_val.png
- reports/figures/roc_logreg_test.png
- reports/figures/pr_logreg_test.png
- reports/figures/calibration_logreg_test.png
