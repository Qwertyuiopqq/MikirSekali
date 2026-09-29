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
