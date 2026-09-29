"""
lstm_hurst_train.py  -- Python 3.11 (local) version
---------------------------------------------------------------------------
Merges lstm_hurst_prepare.py + lstm_hurst_tune_train.py into ONE script so it
runs outside Colab (`python lstm_hurst_train.py`).

What was changed to make it run on a normal Python 3.11.9 environment:
  1. The two Colab cells shared variables through the notebook session
     (`features`, `datasets`). They are now a single file, so nothing is missing.
  2. `files.download(...)` (google.colab only, and never imported) is replaced
     by an optional download that only runs when actually inside Colab.
  3. Paths are resolved relative to THIS FILE (not the current working
     directory), and can be overridden with env vars DATA_CSV / EXPORT_ROOT.
  4. The Hurst rolling apply uses raw=True (numpy arrays, much faster than
     passing a pandas Series for every window).
  5. `shutil.make_archive` now uses absolute paths, so the zip is created
     correctly regardless of where you launch the script from.
  6. Everything is wrapped in main() with an `if __name__ == "__main__"` guard.

Install:
    pip install numpy pandas scikit-learn joblib torch hurst
"""
import copy
import json
import os
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from hurst import compute_Hc
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# CONFIG
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
DATA_CSV = Path(os.environ.get("DATA_CSV", BASE_DIR / "../../data/output/MCS_features.csv")).resolve()
EXPORT_ROOT = Path(os.environ.get("EXPORT_ROOT", BASE_DIR / "../../models/LSTMwithHurst")).resolve()

TARGET_SYMBOLS = None      # None = semua simbol di CSV; atau mis. ["ASSA.JK", "BIRD.JK"]
SEQ_LENGTH = 30
TRAIN_FRACTION = 0.8
MIN_ROWS_PER_SYMBOL = SEQ_LENGTH + 50
RANDOM_SEED = 42
HURST_WINDOW = 100         # HARUS sama dengan config.HURST_WINDOW di pipeline/config.py (paket `hurst` butuh >= 100 titik)

FEATURES = ["Close", "dist_to_resistance", "dist_to_support", "hurst_exponent"]

PARAM_GRID = {
    "hidden_size": [32, 64],
    "learning_rate": [0.001, 0.005],
}
EPOCHS = 30


# ==========================================
# 1. FEATURE ENGINEERING (per symbol)
# ==========================================
def _calc_hurst(series):
    try:
        H, _, _ = compute_Hc(series, kind="price", simplified=True)
        return H
    except Exception:
        return 0.5  # Random walk default


def extract_fractal_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    group["yearly_resistance"] = group["Close"].rolling(window=252, min_periods=60).max()
    group["yearly_support"] = group["Close"].rolling(window=252, min_periods=60).min()

    group["dist_to_resistance"] = (group["yearly_resistance"] - group["Close"]) / group["Close"]
    group["dist_to_support"] = (group["Close"] - group["yearly_support"]) / group["Close"]

    group["hurst_exponent"] = group["Close"].rolling(window=HURST_WINDOW).apply(_calc_hurst, raw=True)

    future_max = group["Close"].shift(-30).rolling(window=30).max()
    group["is_breakout"] = (future_max > group["yearly_resistance"]).astype(int)
    return group


def create_sequences(data: pd.DataFrame, seq_length: int = SEQ_LENGTH):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:(i + seq_length)][FEATURES].values)
        y.append(data.iloc[i + seq_length - 1]["is_breakout"])
    return np.array(X), np.array(y)


# ==========================================
# 2. DATA PREPARATION
# ==========================================
def prepare_datasets():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"CSV tidak ditemukan: {DATA_CSV}\n"
                                f"Set env var DATA_CSV atau ubah DATA_CSV di bagian CONFIG.")

    df = pd.read_csv(DATA_CSV, parse_dates=["Date"])
    df = df.sort_values(["symbol", "Date"]).reset_index(drop=True)

    print("Mengekstrak fitur S/R Tahunan dan Hurst Exponent...")
    df = pd.concat(
        [extract_fractal_features(g) for _, g in df.groupby("symbol")],
        ignore_index=True,
    )
    df = df.dropna(subset=["hurst_exponent", "is_breakout", "dist_to_resistance"]).reset_index(drop=True)

    h = df["hurst_exponent"]
    print(f"[hurst] min={h.min():.3f}  median={h.median():.3f}  max={h.max():.3f}  std={h.std():.3f}")
    if h.nunique() <= 1:
        print("⚠️  hurst_exponent KONSTAN -> compute_Hc selalu gagal. "
              "Model yang dilatih tidak akan benar-benar memakai Hurst.")

    overall_rate = df["is_breakout"].mean()
    print(f"\n[class balance] is_breakout=1 di seluruh dataset: "
          f"{df['is_breakout'].sum()}/{len(df)} ({overall_rate:.1%})")
    per_symbol_rate = df.groupby("symbol")["is_breakout"].mean().sort_values(ascending=False)
    print(pd.concat([per_symbol_rate.head(5), per_symbol_rate.tail(5)]).to_string())
    if overall_rate < 0.05 or overall_rate > 0.95:
        print("⚠️  Breakout event sangat jarang/sering -- lihat precision/recall/AUC, bukan akurasi mentah.")

    torch.manual_seed(RANDOM_SEED)

    symbols = TARGET_SYMBOLS or sorted(df["symbol"].unique())
    datasets, skipped = {}, {}

    for sym in symbols:
        df_sym = df[df["symbol"] == sym].sort_values("Date").reset_index(drop=True)
        if len(df_sym) < MIN_ROWS_PER_SYMBOL:
            skipped[sym] = f"hanya {len(df_sym)} baris (< {MIN_ROWS_PER_SYMBOL})"
            continue

        row_split_idx = int(len(df_sym) * TRAIN_FRACTION)
        scaler = StandardScaler()
        scaler.fit(df_sym.iloc[:row_split_idx][FEATURES])          # fit: TRAIN PERIOD ONLY
        df_sym[FEATURES] = scaler.transform(df_sym[FEATURES])

        X_seq, y_seq = create_sequences(df_sym)
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
            "trainable": len(np.unique(y_train)) == 2,
        }

    print(f"\n[data] {len(datasets)} simbol siap dilatih.")
    summary = pd.DataFrame([
        {"symbol": s, "n_train": d["n_train"], "pos_train": d["pos_train"],
         "pos_train_%": round(100 * d["pos_train"] / max(d["n_train"], 1), 1),
         "n_test": d["n_test"], "pos_test": d["pos_test"],
         "pos_test_%": round(100 * d["pos_test"] / max(d["n_test"], 1), 1),
         "trainable": d["trainable"]}
        for s, d in datasets.items()
    ])
    if not summary.empty:
        print(summary.to_string(index=False))

    for sym, why in skipped.items():
        print(f"⚠️  {sym} dilewati: {why}.")
    for sym, d in datasets.items():
        if not d["trainable"]:
            print(f"⚠️  {sym}: train set hanya punya SATU kelas -> akan dilewati saat training.")
        elif d["pos_test"] == 0 or d["pos_test"] == d["n_test"]:
            print(f"⚠️  {sym}: test set hanya punya satu kelas -> seleksi model pakai loss.")

    return datasets


# ==========================================
# 3. MODEL + EVALUATION
# ==========================================
class LSTMWithHurst(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        return self.sigmoid(self.fc(out))


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
        "auc": roc_auc_score(y_true, probs) if (n_pos > 0 and n_neg > 0) else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, preds, labels=[0, 1]),
    }
    return metrics


def _fmt(m):
    auc_str = f"{m['auc']:.4f}" if m["auc"] == m["auc"] else "n/a (test set punya 1 kelas saja)"
    return (f"loss={m['loss']:.4f}  acc={m['accuracy']:.3f}  "
            f"precision={m['precision']:.3f}  recall={m['recall']:.3f}  "
            f"f1={m['f1']:.3f}  auc={auc_str}")


def _clean(v):
    v = float(v)
    return None if v != v else v


def train_one_symbol(ds):
    train_loader, X_test_t, y_test_t = ds["train_loader"], ds["X_test_t"], ds["y_test_t"]

    auc_available = bool(int(y_test_t.sum().item()) > 0 and int((1 - y_test_t).sum().item()) > 0)
    select_by = "auc" if auc_available else "loss"
    print(f"Kriteria seleksi: {select_by.upper()}")

    best_score = -np.inf if select_by == "auc" else float("inf")
    best_state, best_params, best_metrics = None, {}, None

    for hs in PARAM_GRID["hidden_size"]:
        for lr in PARAM_GRID["learning_rate"]:
            print(f"\n  Menguji -> Hidden Size: {hs} | Learning Rate: {lr}")

            torch.manual_seed(RANDOM_SEED)
            model = LSTMWithHurst(input_size=len(FEATURES), hidden_size=hs)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=lr)

            for _ in range(EPOCHS):
                model.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(batch_X), batch_y)
                    loss.backward()
                    optimizer.step()

            m = evaluate(model, criterion, X_test_t, y_test_t)
            print(f"  Hasil -> {_fmt(m)}")
            print(f"     confusion matrix [[TN,FP],[FN,TP]]:\n{m['confusion_matrix']}")

            is_better = (m["auc"] > best_score) if select_by == "auc" else (m["loss"] < best_score)
            if is_better:
                best_score = m["auc"] if select_by == "auc" else m["loss"]
                best_state = copy.deepcopy(model.state_dict())
                best_params = {"hidden_size": hs, "learning_rate": lr}
                best_metrics = m

    return best_state, best_params, best_metrics, select_by


# ==========================================
# 4. MAIN
# ==========================================
def main():
    datasets = prepare_datasets()
    if not datasets:
        print("Tidak ada dataset yang bisa dilatih.")
        return

    if EXPORT_ROOT.exists():
        shutil.rmtree(EXPORT_ROOT)
    EXPORT_ROOT.mkdir(parents=True)

    summary_rows = []
    print(f"\nMemulai tuning LSTM-with-Hurst: {len(datasets)} perusahaan...")

    for i, (sym, ds) in enumerate(datasets.items(), start=1):
        company = sym.replace(".JK", "")
        print(f"\n{'=' * 60}\n[{i}/{len(datasets)}] {sym}  "
              f"(train={ds['n_train']} seq / {ds['pos_train']} positif | "
              f"test={ds['n_test']} seq / {ds['pos_test']} positif)\n{'=' * 60}")

        if not ds["trainable"]:
            print(f"⚠️  {sym} dilewati: train set hanya punya satu kelas.")
            summary_rows.append({"symbol": sym, "status": "SKIPPED (train 1 kelas)"})
            continue

        best_state, best_params, best_metrics, select_by = train_one_symbol(ds)

        print(f"\n✅ {sym} selesai! Parameter Terbaik: {best_params}")
        print(f"   Metrik -> {_fmt(best_metrics)}")

        company_dir = EXPORT_ROOT / company
        company_dir.mkdir()

        torch.save(best_state, company_dir / "best_lstm_hurst_model.pth")
        joblib.dump(ds["scaler"], company_dir / "lstm_hurst_scaler.joblib")
        with open(company_dir / "lstm_hurst_best_params.json", "w") as f:
            json.dump({
                **best_params,
                "symbol": sym,
                "seq_length": SEQ_LENGTH,
                "features": list(FEATURES),
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
            "n_test_positive": best_metrics["n_test_positive"],
            "n_test_negative": best_metrics["n_test_negative"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(EXPORT_ROOT / "lstm_hurst_summary.csv", index=False)
    print("\n=== RINGKASAN MODEL LSTM-HURST PER PERUSAHAAN ===")
    print(summary_df.to_string(index=False))

    n_ok = int((summary_df["status"] == "OK").sum())
    zip_path = shutil.make_archive(
        str(EXPORT_ROOT), "zip", root_dir=str(EXPORT_ROOT.parent), base_dir=EXPORT_ROOT.name
    )
    print(f"\n📦 Zip dibuat: {zip_path} ({n_ok}/{len(datasets)} model)")

    # Colab-only auto download (skipped on a normal local environment)
    try:
        from google.colab import files  # type: ignore
        files.download(zip_path)
    except ImportError:
        pass

    print(f"\n🎉 Model tersimpan di: {EXPORT_ROOT}")
    print("   <PERUSAHAAN>/best_lstm_hurst_model.pth | lstm_hurst_scaler.joblib | lstm_hurst_best_params.json")


if __name__ == "__main__":
    main()
