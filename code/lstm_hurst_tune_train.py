"""
lstm_hurst_tune_train.py  (Colab cell 3 -- run after lstm_hurst_prepare.py)
-----------------------------------------------------------------------------
Hyperparameter grid search + final evaluation + export for the
LSTM-with-Hurst breakout model -- ONE MODEL PER COMPANY.

For every symbol in `datasets` (from lstm_hurst_prepare.py) this runs the
same grid search as before on that symbol's own train/test sequences,
keeps that symbol's best model, and exports it under its own folder. One
zip is downloaded at the end (instead of 3 files x N companies):

    LSTMwithHurst.zip
      LSTMwithHurst/
        ASSA/  best_lstm_hurst_model.pth
               lstm_hurst_scaler.joblib
               lstm_hurst_best_params.json
        BIRD/  ...
        lstm_hurst_summary.csv         (metrics of every company's best model)

>>> Unzip it INTO the project's `models/` folder so you get
>>> models/LSTMwithHurst/<COMPANY>/...  -- the path pipeline/config.py
>>> (`lstm_hurst_model_path`) already points to.

Changes vs. the single-symbol version:
  1. The grid search / evaluation / selection code is unchanged; it just runs
     once per symbol (fresh torch seed per symbol so results are reproducible).
  2. Symbols whose train set has a single class (`trainable=False` in
     lstm_hurst_prepare.py) are skipped -- no constant "model" is exported;
     the pipeline then leaves that company's breakout_probability empty.
  3. best_params.json now also stores seq_length, symbol, selection
     criterion and the test metrics next to hidden_size/learning_rate
     (the pipeline only reads hidden_size and seq_length).

Earlier changes kept: evaluation reports accuracy/precision/recall/F1/AUC +
confusion matrix (not just BCE loss); selection prefers AUC when the test set
has both classes, otherwise falls back to loss; scaler + best_params are
exported next to the weights (inference needs the exact training scaler).
"""
import copy
import json
import os
import shutil

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from google.colab import files
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

# `features`, `datasets` (and SEQ_LENGTH, RANDOM_SEED) come from
# lstm_hurst_prepare.py, run earlier in this same Colab session.

# ==========================================
# 1. ARSITEKTUR MODEL (Sesuai Paradigma Fraktal)
# ==========================================
class LSTMWithHurst(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        super(LSTMWithHurst, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        out = self.dropout(last_step)
        out = self.fc(out)
        return self.sigmoid(out)


# ==========================================
# 2. EVALUASI KLASIFIKASI (bukan cuma loss)
# ==========================================
def evaluate(model, criterion, X_test_t, y_test_t):
    model.eval()
    with torch.no_grad():
        test_probs_t = model(X_test_t)
        loss = criterion(test_probs_t, y_test_t).item()

    probs = test_probs_t.numpy().ravel()
    y_true = y_test_t.numpy().ravel().astype(int)
    preds = (probs >= 0.5).astype(int)

    n_pos, n_neg = int(y_true.sum()), int((1 - y_true).sum())
    metrics = {
        "loss": loss,
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "n_test_positive": n_pos,
        "n_test_negative": n_neg,
    }
    # AUC is undefined with only one class present in y_true -- don't
    # pretend it means something in that case.
    if n_pos > 0 and n_neg > 0:
        metrics["auc"] = roc_auc_score(y_true, probs)
    else:
        metrics["auc"] = float("nan")

    metrics["confusion_matrix"] = confusion_matrix(y_true, preds, labels=[0, 1])
    return metrics


def _fmt(m):
    auc_str = f"{m['auc']:.4f}" if m["auc"] == m["auc"] else "n/a (test set punya 1 kelas saja)"
    return (f"loss={m['loss']:.4f}  acc={m['accuracy']:.3f}  "
            f"precision={m['precision']:.3f}  recall={m['recall']:.3f}  "
            f"f1={m['f1']:.3f}  auc={auc_str}")


# ==========================================
# 3. HYPERPARAMETER TUNING (Grid Search) -- PER PERUSAHAAN
# ==========================================
param_grid = {
    "hidden_size": [32, 64],
    "learning_rate": [0.001, 0.005],
}
EPOCHS = 30
EXPORT_ROOT = "LSTMwithHurst"

input_size = len(features)  # dari lstm_hurst_prepare.py


def _clean(v):
    """JSON-safe number (NaN -> None, numpy scalars -> python)."""
    v = float(v)
    return None if v != v else v


def train_one_symbol(sym, ds):
    """Grid search for ONE symbol. Returns (best_state, best_params, best_metrics, select_by) or None."""
    train_loader, X_test_t, y_test_t = ds["train_loader"], ds["X_test_t"], ds["y_test_t"]

    auc_available = bool(int(y_test_t.sum().item()) > 0 and int((1 - y_test_t).sum().item()) > 0)
    select_by = "auc" if auc_available else "loss"
    print(f"Kriteria seleksi: {select_by.upper()} "
          f"({'test set punya kedua kelas' if auc_available else 'test set cuma 1 kelas -> AUC tidak valid, pakai loss'})")

    best_score = -np.inf if select_by == "auc" else float("inf")
    best_state, best_params, best_metrics = None, {}, None

    for hs in param_grid["hidden_size"]:
        for lr in param_grid["learning_rate"]:
            print(f"\n  Menguji Parameter -> Hidden Size: {hs} | Learning Rate: {lr}")

            torch.manual_seed(RANDOM_SEED)
            model = LSTMWithHurst(input_size=input_size, hidden_size=hs)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            for epoch in range(EPOCHS):
                model.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    predictions = model(batch_X)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    optimizer.step()

            m = evaluate(model, criterion, X_test_t, y_test_t)
            print(f"  Hasil Evaluasi -> {_fmt(m)}")
            print(f"     confusion matrix [[TN,FP],[FN,TP]]:\n{m['confusion_matrix']}")

            is_better = (m["auc"] > best_score) if select_by == "auc" else (m["loss"] < best_score)
            if is_better:
                best_score = m["auc"] if select_by == "auc" else m["loss"]
                best_state = copy.deepcopy(model.state_dict())
                best_params = {"hidden_size": hs, "learning_rate": lr}
                best_metrics = m

    return best_state, best_params, best_metrics, select_by


# ==========================================
# 4. LATIH SEMUA PERUSAHAAN + SIMPAN PER FOLDER
# ==========================================
if os.path.exists(EXPORT_ROOT):
    shutil.rmtree(EXPORT_ROOT)  # jangan sisakan model dari run sebelumnya
os.makedirs(EXPORT_ROOT)

summary_rows = []
print(f"\nMemulai Hyperparameter Tuning untuk LSTM-with-Hurst: {len(datasets)} perusahaan...")

for i, (sym, ds) in enumerate(datasets.items(), start=1):
    company = sym.replace(".JK", "")
    print(f"\n{'=' * 60}\n[{i}/{len(datasets)}] {sym}  "
          f"(train={ds['n_train']} seq / {ds['pos_train']} positif | "
          f"test={ds['n_test']} seq / {ds['pos_test']} positif)\n{'=' * 60}")

    if not ds["trainable"]:
        print(f"⚠️  {sym} dilewati: train set hanya punya satu kelas.")
        summary_rows.append({"symbol": sym, "status": "SKIPPED (train 1 kelas)"})
        continue

    best_state, best_params, best_metrics, select_by = train_one_symbol(sym, ds)

    print(f"\n✅ {sym} selesai! Parameter Terbaik: {best_params}")
    print(f"   Metrik model terbaik -> {_fmt(best_metrics)}")
    if select_by == "loss":
        print("   ⚠️  Dipilih berdasarkan loss (bukan AUC) karena test set hanya punya satu kelas.")

    company_dir = os.path.join(EXPORT_ROOT, company)
    os.makedirs(company_dir)

    torch.save(best_state, os.path.join(company_dir, "best_lstm_hurst_model.pth"))
    joblib.dump(ds["scaler"], os.path.join(company_dir, "lstm_hurst_scaler.joblib"))  # scaler MILIK simbol ini
    with open(os.path.join(company_dir, "lstm_hurst_best_params.json"), "w") as f:
        json.dump({
            **best_params,
            "symbol": sym,
            "seq_length": SEQ_LENGTH,
            "features": list(features),
            "select_by": select_by,
            "test_auc": _clean(best_metrics["auc"]),
            "test_f1": _clean(best_metrics["f1"]),
            "n_test_positive": best_metrics["n_test_positive"],
            "n_test_negative": best_metrics["n_test_negative"],
        }, f, indent=2)

    summary_rows.append({
        "symbol": sym, "status": "OK", **best_params, "select_by": select_by,
        "loss": round(best_metrics["loss"], 4), "accuracy": round(best_metrics["accuracy"], 3),
        "precision": round(best_metrics["precision"], 3), "recall": round(best_metrics["recall"], 3),
        "f1": round(best_metrics["f1"], 3),
        "auc": None if best_metrics["auc"] != best_metrics["auc"] else round(best_metrics["auc"], 4),
        "n_test_positive": best_metrics["n_test_positive"], "n_test_negative": best_metrics["n_test_negative"],
    })

# ==========================================
# 5. RINGKASAN + SATU ZIP UNTUK DIUNDUH
# ==========================================
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(EXPORT_ROOT, "lstm_hurst_summary.csv"), index=False)
print("\n=== RINGKASAN MODEL LSTM-HURST PER PERUSAHAAN (test set masing-masing) ===")
print(summary_df.to_string(index=False))

n_ok = int((summary_df["status"] == "OK").sum())
zip_path = shutil.make_archive(EXPORT_ROOT, "zip", root_dir=".", base_dir=EXPORT_ROOT)
print(f"\n📥 Mengunduh {zip_path} ({n_ok}/{len(datasets)} model)...")
files.download(zip_path)

print("\n🎉 Ekstrak zip ke folder `models/` proyek, sehingga strukturnya:")
print("   models/LSTMwithHurst/<PERUSAHAAN>/best_lstm_hurst_model.pth")
print("   models/LSTMwithHurst/<PERUSAHAAN>/lstm_hurst_scaler.joblib")
print("   models/LSTMwithHurst/<PERUSAHAAN>/lstm_hurst_best_params.json")
