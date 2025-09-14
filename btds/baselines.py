import argparse, json, os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, average_precision_score, brier_score_loss
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--target', default='Diabetes_binary')
    ap.add_argument('--out_dir', default='reports')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.data_path)
    if args.target not in df.columns:
        args.target = 'Diabetes_binary' if 'Diabetes_binary' in df.columns else 'Diabetes_012'
    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

    # Logistic Regression
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=200, class_weight='balanced'))
    ])
    lr.fit(X_tr, y_tr)
    lr_acc = lr.score(X_te, y_te)
    if y.nunique()==2:
        proba = lr.predict_proba(X_te)[:,1]
        lr_auc = roc_auc_score(y_te, proba)
        lr_ap  = average_precision_score(y_te, proba)
        lr_brier = brier_score_loss(y_te, proba)
    else:
        lr_auc = lr_ap = lr_brier = None
    with open(os.path.join(args.out_dir, 'baseline_logreg_metrics.json'), 'w') as f:
        json.dump({
            'accuracy': float(lr_acc),
            'auc_roc': float(lr_auc) if lr_auc is not None else None,
            'avg_precision': float(lr_ap) if lr_ap is not None else None,
            'brier': float(lr_brier) if lr_brier is not None else None
        }, f, indent=2)

    # HistGradientBoosting
    hgb = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.1)
    hgb.fit(X_tr, y_tr)
    hgb_acc = hgb.score(X_te, y_te)
    if y.nunique()==2:
        proba = hgb.predict_proba(X_te)[:,1]
        hgb_auc = roc_auc_score(y_te, proba)
        hgb_ap  = average_precision_score(y_te, proba)
        hgb_brier = brier_score_loss(y_te, proba)
    else:
        hgb_auc = hgb_ap = hgb_brier = None
    with open(os.path.join(args.out_dir, 'baseline_hgb_metrics.json'), 'w') as f:
        json.dump({
            'accuracy': float(hgb_acc),
            'auc_roc': float(hgb_auc) if hgb_auc is not None else None,
            'avg_precision': float(hgb_ap) if hgb_ap is not None else None,
            'brier': float(hgb_brier) if hgb_brier is not None else None
        }, f, indent=2)

if __name__ == '__main__': main()
