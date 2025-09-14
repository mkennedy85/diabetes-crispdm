import numpy as np
from sklearn.calibration import calibration_curve
import argparse, os, json, numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay,
    roc_auc_score, average_precision_score
)
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim:int, hidden:int=256, p:float=0.25, out_dim:int=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x): return self.net(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--target', default='Diabetes_binary')
    ap.add_argument('--out_dir', default='reports')
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, 'figures'), exist_ok=True)

    df = pd.read_csv(args.data_path)
    if args.target not in df.columns:
        args.target = 'Diabetes_binary' if 'Diabetes_binary' in df.columns else 'Diabetes_012'
    X = df.drop(columns=[args.target]).values.astype('float32')
    y = df[args.target].values

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype('float32')
    X_va = scaler.transform(X_va).astype('float32')
    X_te = scaler.transform(X_te).astype('float32')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLP(in_dim=X.shape[1], out_dim=int(np.unique(y).shape[0])).to(device)
    model.load_state_dict(torch.load(os.path.join(args.out_dir, 'mlp_best.pt'), map_location=device))
    model.eval()

    import numpy as np
    with torch.no_grad():
        logits = model(torch.tensor(X_te).to(device)).cpu().numpy()
    preds = logits.argmax(1)
    # Confusion matrix
    from sklearn.metrics import classification_report
    cm = confusion_matrix(y_te, preds)
    ConfusionMatrixDisplay(cm).plot(values_format="d")
    plt.title("MLP — Confusion Matrix (Test)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'figures', 'cm_mlp_test.png')); plt.close()

    # ROC/PR if binary
    if len(np.unique(y))==2:
        from scipy.special import softmax
        proba = softmax(logits, axis=1)[:,1]
        RocCurveDisplay.from_predictions(y_te, proba)
        plt.title("MLP — ROC (Test)")
        plt.savefig(os.path.join(args.out_dir, 'figures', 'roc_mlp_test.png')); plt.close()

        PrecisionRecallDisplay.from_predictions(y_te, proba)
        plt.title("MLP — PR (Test)")
        plt.savefig(os.path.join(args.out_dir, 'figures', 'pr_mlp_test.png')); plt.close()

        # Calibration curve
        frac_pos, mean_pred = calibration_curve(y_te, proba, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o")
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Fraction of positives")
        plt.title("MLP — Calibration (Test)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'figures', 'calibration_mlp_test.png')); plt.close()

    # Save a summary json
    rep = classification_report(y_te, preds, output_dict=True)
    with open(os.path.join(args.out_dir, 'mlp_eval_report.json'), 'w') as f:
        json.dump(rep, f, indent=2)

if __name__ == '__main__': main()
