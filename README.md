# Diabetes (BRFSS 2015) — CRISP-DM (Scikit + PyTorch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mkennedy85/diabetes-crispdm/blob/main/notebooks/eda_and_model.ipynb)

This project implements a textbook-quality **CRISP-DM** workflow on BRFSS 2015 diabetes indicators with:
- Scikit-learn baselines (LogReg, HistGradientBoosting)
- A compact **PyTorch MLP** baseline suitable for **8 GB RAM** environments
- Reproducible splits, metrics exports, and ready-to-embed figures

### Medium Blog Post

[📊 Predicting Diabetes from Health Indicators Using CRISP-DM](https://medium.com/@michaellkennedy/predicting-diabetes-from-health-indicators-using-crisp-dm-c824c5700e26)

## Quickstart
```bash
pip install -r requirements.txt
# put your CSV in data/
make baselines
make train
make eval
```

## Repo Structure
- `notebooks/eda_and_model.ipynb` — CRISP-DM EDA + visuals
- `btds/baselines.py` — scikit baselines + metrics JSON
- `btds/train.py` — PyTorch MLP with AMP, early stopping, class weights
- `btds/eval.py` — MLP eval: confusion matrix, ROC/PR, calibration
- `reports/` — figures, metrics, and `report.md` (CRISP-DM write-up)

## Data
## Data

In Colab, upload the dataset directly via the **first cell** in the notebook.  
The file will be placed into `data/`, and the notebook will read from there.

Expected filenames (either is fine):
- `data/diabetes_binary_health_indicators_BRFSS2015.csv`
- `data/diabetes_012_health_indicators_BRFSS2015.csv`


### Colab note
If you open the notebook via the Colab badge, run the **first cell** titled *Repo Setup (Colab)*.  
It will clone the repo and set the working directory so that `btds` can be imported (needed for baselines/train/eval).
