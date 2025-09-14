import argparse, json, os, numpy as np, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report

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
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch_size', type=int, default=512)   # 8GB-friendly
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--patience', type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.data_path)
    if args.target not in df.columns:
        args.target = 'Diabetes_binary' if 'Diabetes_binary' in df.columns else 'Diabetes_012'
    X = df.drop(columns=[args.target]).values.astype('float32')
    y = df[args.target].values

    # splits
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

    # scaling
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype('float32')
    X_va = scaler.transform(X_va).astype('float32')
    X_te = scaler.transform(X_te).astype('float32')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tr_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    va_ds = TensorDataset(torch.tensor(X_va), torch.tensor(y_va))
    te_ds = TensorDataset(torch.tensor(X_te), torch.tensor(y_te))

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, num_workers=0)
    te_dl = DataLoader(te_ds, batch_size=args.batch_size, num_workers=0)

    n_classes = int(np.unique(y).shape[0])
    model = MLP(in_dim=X.shape[1], out_dim=n_classes).to(device)

    # class weights for imbalance
    uniq, counts = np.unique(y_tr, return_counts=True)
    weights = counts.sum()/ (counts * len(counts))
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor) if n_classes>1 else nn.BCEWithLogitsLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(10,args.epochs))
    scaler_amp = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_f1, patience = -1.0, args.patience
    os.makedirs(args.out_dir, exist_ok=True)

    for ep in range(1, args.epochs+1):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device).long()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            scaler_amp.scale(loss).backward()
            scaler_amp.step(opt)
            scaler_amp.update()
        sched.step()

        # val
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in va_dl:
                xb = xb.to(device)
                logits = model(xb)
                preds.extend(logits.argmax(1).cpu().numpy())
                trues.extend(yb.numpy())
        val_f1 = f1_score(trues, preds, average='macro')
        print(f"epoch {ep:02d} val_macro_f1={val_f1:.4f} lr={sched.get_last_lr()[0]:.2e}")

        if val_f1 > best_f1:
            best_f1 = val_f1; patience = args.patience
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'mlp_best.pt'))
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping.")
                break

    # Test
    model.load_state_dict(torch.load(os.path.join(args.out_dir, 'mlp_best.pt'), map_location=device))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in te_dl:
            xb = xb.to(device)
            logits = model(xb)
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(yb.numpy())
    acc = accuracy_score(trues, preds)
    f1  = f1_score(trues, preds, average='macro')
    rep = classification_report(trues, preds, output_dict=True)
    with open(os.path.join(args.out_dir, 'mlp_metrics.json'), 'w') as f:
        json.dump({"accuracy": acc, "macro_f1": f1, "report": rep}, f, indent=2)
    print("MLP test_acc", acc, "macro_f1", f1)

if __name__ == '__main__': main()
