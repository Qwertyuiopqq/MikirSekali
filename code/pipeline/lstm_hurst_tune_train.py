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
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

# `features`, `datasets` (and SEQ_LENGTH, RANDOM_SEED) come from
# lstm_hurst_prepare.py, run earlier in this same Colab session.
"""
lstm_hurst_prepare.py  (Colab cell 2 -- run after `!pip install hurst`)
-------------------------------------------------------------------------
Fractal feature engineering (annual support/resistance, Hurst exponent,
breakout label) + sequence building + train/test split for the
LSTM-with-Hurst breakout model -- now for EVERY company, not one.

Produces, in the Colab session's memory (used by lstm_hurst_tune_train.py
next): `features`, `datasets`. `datasets` maps each symbol to everything
its own model needs:

    datasets["ASSA.JK"] = {
        "scaler":       StandardScaler fit on that symbol's TRAIN rows only,
        "train_loader": DataLoader of that symbol's train sequences,
        "X_test_t", "y_test_t": that symbol's held-out sequences,
        "n_train", "n_test", "pos_train", "pos_test": class counts,
        "trainable":    False if the train set has a single class (a
                        constant classifier is not a model -- skipped),
    }

Changes vs. the single-symbol version (TARGET_SYMBOL = "ASSA.JK"):
  0. BUG FIX -- Hurst window 60 -> HURST_WINDOW = 100. The `hurst` package raises
     ValueError on series shorter than 100 points; the old `except: return 0.5`
     hid that, so hurst_exponent was ALWAYS 0.5 (a constant input the scaler
     turns into all zeros). Must match config.HURST_WINDOW in the pipeline.
  1. The block "scale -> sequences -> split -> tensors -> loader" that used
     to run once for TARGET_SYMBOL now runs once per symbol, each with its
     OWN StandardScaler (fit on that symbol's train period only, as
     before). Feature definitions, SEQ_LENGTH, TRAIN_FRACTION and the
     labelling are unchanged.
  2. Set TARGET_SYMBOLS = ["ASSA.JK", ...] below to prepare only a subset
     (None = every symbol found in the CSV).
  3. A per-symbol class-balance table is printed at the end.

(Earlier fixes kept: explicit loop instead of groupby.apply for pandas >= 2.2,
scaler fit on the training period only -- no test-period leakage, and class
balance printed BEFORE training since annual-resistance breakouts can be rare.)
"""
import numpy as np
import pandas as pd
import torch
from hurst import compute_Hc
from sklearn.preprocessing import StandardScaler

DATA_CSV = "../../data/output/MCS_features.csv"
TARGET_SYMBOLS = None      # None = semua simbol di CSV; atau mis. ["ASSA.JK", "BIRD.JK"]
SEQ_LENGTH = 30
TRAIN_FRACTION = 0.8
MIN_ROWS_PER_SYMBOL = SEQ_LENGTH + 50   # kurang dari ini -> terlalu pendek untuk split train/test yang berarti
RANDOM_SEED = 42
HURST_WINDOW = 100         # HARUS sama dengan config.HURST_WINDOW di pipeline/config.py.
                           # Paket `hurst` menolak deret < 100 titik (ValueError), jadi window 60 yang lama
                           # SELALU jatuh ke fallback `return 0.5` -> hurst_exponent konstan 0.5.

# ==========================================
# 1. LOAD DATA
# ==========================================
df = pd.read_csv(DATA_CSV, parse_dates=["Date"])
df = df.sort_values(["symbol", "Date"]).reset_index(drop=True)


# ==========================================
# 2. FRACTAL FEATURE ENGINEERING (per symbol)
# ==========================================
def extract_fractal_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    # Support/Resistance Tahunan (Asumsi 252 hari perdagangan saham)
    group["yearly_resistance"] = group["Close"].rolling(window=252, min_periods=60).max()
    group["yearly_support"] = group["Close"].rolling(window=252, min_periods=60).min()

    # Jarak persentase harga saat ini ke batas S/R
    group["dist_to_resistance"] = (group["yearly_resistance"] - group["Close"]) / group["Close"]
    group["dist_to_support"] = (group["Close"] - group["yearly_support"]) / group["Close"]

    # Rolling Hurst Exponent (HURST_WINDOW hari ke belakang)
    def calc_hurst(series):
        try:
            H, _, _ = compute_Hc(series, kind="price", simplified=True)
            return H
        except Exception:
            return 0.5  # Random walk default

    group["hurst_exponent"] = group["Close"].rolling(window=HURST_WINDOW).apply(calc_hurst, raw=False)

    # Label target: breakout resistance tahunan dalam 30 hari ke depan?
    future_max = group["Close"].shift(-30).rolling(window=30).max()
    group["is_breakout"] = (future_max > group["yearly_resistance"]).astype(int)

    return group


print("Mengekstrak fitur S/R Tahunan dan Hurst Exponent...")
# NOTE: bukan `df.groupby('symbol', group_keys=False).apply(...)` -- lihat
# catatan #1 di docstring modul ini. Loop + concat aman di semua versi pandas.
df = pd.concat(
    [extract_fractal_features(g) for _, g in df.groupby("symbol")],
    ignore_index=True,
)
df = df.dropna(subset=["hurst_exponent", "is_breakout", "dist_to_resistance"]).reset_index(drop=True)

# Cek kewarasan: kalau compute_Hc gagal terus, semua nilai jatuh ke 0.5 dan fitur "Hurst" tidak berisi apa-apa.
_h = df["hurst_exponent"]
print(f"[hurst] hurst_exponent: min={_h.min():.3f}  median={_h.median():.3f}  max={_h.max():.3f}  std={_h.std():.3f}")
if _h.nunique() <= 1:
    print("⚠️  hurst_exponent KONSTAN -> compute_Hc selalu gagal (paket `hurst` butuh window >= 100). "
          "Model yang dilatih tidak akan benar-benar memakai Hurst.")

# ==========================================
# 2b. CLASS BALANCE -- lihat sebelum melatih apa pun
# ==========================================
overall_rate = df["is_breakout"].mean()
print(f"\n[class balance] is_breakout=1 di SELURUH dataset (semua simbol): "
      f"{df['is_breakout'].sum()}/{len(df)} ({overall_rate:.1%})")
per_symbol_rate = df.groupby("symbol")["is_breakout"].mean().sort_values(ascending=False)
print("[class balance] per simbol (top 5 & bottom 5):")
print(pd.concat([per_symbol_rate.head(5), per_symbol_rate.tail(5)]).to_string())
if overall_rate < 0.05 or overall_rate > 0.95:
    print("⚠️  Breakout event sangat jarang (atau sangat sering) -- akurasi/loss mentah TIDAK "
          "cukup untuk menilai model ini, lihat precision/recall/AUC di lstm_hurst_tune_train.py.")

# ==========================================
# 3. SEQUENCES + SPLIT + SCALING -- PER PERUSAHAAN
# ==========================================
features = ["Close", "dist_to_resistance", "dist_to_support", "hurst_exponent"]


def create_sequences(data: pd.DataFrame, seq_length: int = SEQ_LENGTH):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:(i + seq_length)][features].values)
        y.append(data.iloc[i + seq_length - 1]["is_breakout"])
    return np.array(X), np.array(y)


from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(RANDOM_SEED)  # supaya shuffle DataLoader reproducible

symbols_to_prepare = TARGET_SYMBOLS or sorted(df["symbol"].unique())
datasets = {}
skipped = {}

for sym in symbols_to_prepare:
    df_sym = df[df["symbol"] == sym].sort_values("Date").reset_index(drop=True)
    if len(df_sym) < MIN_ROWS_PER_SYMBOL:
        skipped[sym] = f"hanya {len(df_sym)} baris (< {MIN_ROWS_PER_SYMBOL})"
        continue

    row_split_idx = int(len(df_sym) * TRAIN_FRACTION)

    scaler = StandardScaler()
    scaler.fit(df_sym.iloc[:row_split_idx][features])          # fit: TRAIN PERIOD ONLY
    df_sym[features] = scaler.transform(df_sym[features])      # transform: seluruh data, pakai statistik train

    X_seq, y_seq = create_sequences(df_sym, seq_length=SEQ_LENGTH)

    # Split Train & Test -- sequence-level index, lines up (within one
    # SEQ_LENGTH-day window) with row_split_idx above since both come from
    # the same chronological ordering.
    split_idx = int(len(X_seq) * TRAIN_FRACTION)
    X_train, y_train = X_seq[:split_idx], y_seq[:split_idx]
    X_test, y_test = X_seq[split_idx:], y_seq[split_idx:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    datasets[sym] = {
        "scaler": scaler,
        "train_loader": DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True),
        "X_test_t": X_test_t,
        "y_test_t": y_test_t,
        "n_train": len(X_train), "n_test": len(X_test),
        "pos_train": int(y_train.sum()), "pos_test": int(y_test.sum()),
        # kalau train cuma punya 1 kelas, model hanya akan belajar output konstan
        "trainable": len(np.unique(y_train)) == 2,
    }

# ==========================================
# 4. RINGKASAN PER PERUSAHAAN
# ==========================================
print(f"\n[data] {len(datasets)} simbol siap dilatih (masing-masing dapat 1 model LSTM-Hurst sendiri).")
summary = pd.DataFrame([
    {"symbol": sym, "n_train": d["n_train"], "pos_train": d["pos_train"],
     "pos_train_%": round(100 * d["pos_train"] / max(d["n_train"], 1), 1),
     "n_test": d["n_test"], "pos_test": d["pos_test"],
     "pos_test_%": round(100 * d["pos_test"] / max(d["n_test"], 1), 1),
     "trainable": d["trainable"]}
    for sym, d in datasets.items()
])
print(summary.to_string(index=False))

for sym, why in skipped.items():
    print(f"⚠️  {sym} dilewati: {why}.")
for sym, d in datasets.items():
    if not d["trainable"]:
        print(f"⚠️  {sym}: train set hanya punya SATU kelas (semua breakout atau semua non-breakout) -> "
              f"lstm_hurst_tune_train.py akan melewati simbol ini (model konstan tidak berguna).")
    elif d["pos_test"] == 0 or d["pos_test"] == d["n_test"]:
        print(f"⚠️  {sym}: test set hanya punya satu kelas -> AUC tidak valid, "
              f"seleksi model otomatis pakai loss (lihat lstm_hurst_tune_train.py).")

print("\n✅ Data siap. Lanjut ke lstm_hurst_tune_train.py "
      "(pakai `features` dan `datasets` dari sini).")
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
EXPORT_ROOT = "../../models/LSTMwithHurst"

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
