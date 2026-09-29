"""
train_xgboost_transfer.py
--------------------------
Colab training script for the base + per-symbol fine-tuned XGBoost volume
models. Reads MCS_features.csv (Stage 2 output of mcs_pipeline), trains a
pooled base model across every symbol, then fine-tunes one model per symbol
via transfer learning (`xgb_model=base_model_path`).

Changes vs. the original version:
  1. No more blanket `df.fillna(0)`. NaN is handled per-column based on
     what it actually means (see IMPUTASI section) instead of turning
     every missing value -- including a missing macro index price -- into
     a literal 0.
  2. Target is trained in log1p space (`np.log1p`) and inverted with
     `np.expm1` before any evaluation/printing, so a single global model
     can handle symbols that differ by ~1000x in volume scale (e.g.
     BLOG.JK ~276K/day vs GIAA.JK ~186M/day) without the loss being
     dominated by the highest-volume symbol.
  3. Evaluation now reports MAE, WMAPE and SMAPE (matching
     testing/metrics.py) instead of nothing -- the original script never
     computed an aggregate accuracy number, just printed a head().

IMPORTANT -- this changes the model's output space. `pipeline/xgboost_predictor.py`
must call `np.expm1(...)` on `model.predict(...)` to get back to real
volume, or every inference will be wrong (it'll return log-scale numbers,
e.g. ~14-20 instead of millions). That companion fix is applied separately
in this same codebase -- if you retrain with this script, make sure that
fix is in place before running inference.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from google.colab import files

# ==========================================
# 1. UNGGAH DATASET (INTERAKTIF COLAB)
# ==========================================
# Sumber data: MCS_features.csv -- output Stage 2 dari mcs_pipeline
# (raw spine + sentiment/macro/fundamental, siap dipakai XGBoost).
df = pd.read_csv("MCS_features.csv")
print("✅ Dataset berhasil dibaca!")

# ==========================================
# 2. KONTRAK FITUR — HARUS SAMA PERSIS DENGAN mcs_pipeline/config.py
# ==========================================
# xgboost_predictor.py memanggil model.predict(df[XGBOOST_FEATURES]) dengan
# daftar kolom yang TETAP (19 kolom). Jaga daftar ini tetap sinkron dengan
# config.XGBOOST_FEATURES supaya model yang dilatih di sini cocok saat
# dipakai untuk inference di pipeline lokal.
XGBOOST_FEATURES = [
    'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends',
    'Stock Splits', 'day_of_week', 'is_weekend', 'is_dividend',
    'is_stock_split', 'daily_return_pct', 'MA_7_Close', 'MA_30_Close',
    'MA_7_Volume', 'volatility_7d', 'daily_news_sentiment',
    'daily_news_count', 'idx_macro_close'
]
TARGET_COLUMN = 'target_volume_T_plus_1'

# ==========================================
# 2b. IMPUTASI NaN — PER KOLOM, BUKAN BLANKET fillna(0)
# ==========================================
# Kenapa bukan df.fillna(0) global:
#   - XGBoost's DMatrix natively supports NaN (default missing=np.nan) and
#     will learn an optimal split direction for genuinely missing values.
#     fillna(0) was never actually required to avoid a DMatrix error.
#   - target_volume_T_plus_1: BUKAN untuk diisi 0. Baris tanpa target valid
#     (mis. hari terakhir per simbol, shift(-1) menghasilkan NaN) harus
#     DIBUANG, bukan diberi label palsu "volume besok = 0".
#   - idx_macro_close, ROE/ROA/gross_margin/debt_to_equity/forward_pe:
#     ini nilai level (harga indeks, rasio keuangan), bukan sesuatu yang
#     "wajar" bernilai 0. enrichment.py sudah ffill() idx_macro_close
#     setelah merge, tapi baris paling awal (sebelum data makro pertama
#     tersedia) masih bisa NaN karena ffill tidak bisa mengisi ke belakang.
#     Kita isi per-simbol dengan ffill lalu bfill (bukan 0).
#   - daily_return_pct, volatility_7d: NaN di sini SECARA WAJAR terjadi di
#     hari pertama per simbol (pct_change tidak punya nilai sebelumnya;
#     rolling-std butuh >1 titik data). 0 di sini memang default yang
#     masuk akal ("belum ada return/volatilitas teramati"), jadi TETAP
#     diisi 0 -- tapi hanya untuk dua kolom ini, bukan seluruh dataframe.

n_before = len(df)
df = df.dropna(subset=[TARGET_COLUMN]).copy()
n_dropped = n_before - len(df)
if n_dropped:
    print(f"[imputasi] {n_dropped} baris dibuang karena target ({TARGET_COLUMN}) NaN "
          f"(bukan diisi 0 -- itu akan memalsukan label).")

_level_cols = [c for c in
               ['idx_macro_close', 'ROE', 'ROA', 'gross_margin', 'debt_to_equity', 'forward_pe']
               if c in df.columns]
if _level_cols:
    df[_level_cols] = df.groupby('symbol')[_level_cols].transform(lambda s: s.ffill().bfill())
    still_missing = df[_level_cols].isna().sum()
    still_missing = still_missing[still_missing > 0]
    if not still_missing.empty:
        print(f"[imputasi] Kolom level yang masih ada NaN setelah ffill/bfill per simbol "
              f"(dibiarkan NaN, XGBoost akan menanganinya secara native): {still_missing.to_dict()}")

for _col in ['daily_return_pct', 'volatility_7d']:
    if _col in df.columns:
        n_na = int(df[_col].isna().sum())
        if n_na:
            print(f"[imputasi] {_col}: {n_na} NaN (hari pertama per simbol) -> diisi 0.")
        df[_col] = df[_col].fillna(0)

print(f"✅ Imputasi selesai. Sisa NaN di fitur akan ditangani native oleh XGBoost DMatrix.")

# ==========================================
# 3. DATA BASE MODEL (GABUNGAN SEMUA PERUSAHAAN)
# ==========================================
X_base = df[XGBOOST_FEATURES]
y_base = df[TARGET_COLUMN]

# Log1p transform: pool berisi simbol dengan skala volume yang berbeda
# ~1000x (mis. BLOG.JK vs GIAA.JK). Tanpa ini, base model didominasi
# skala simbol dengan volume terbesar dan nyaris buta terhadap simbol
# bervolume tipis.
y_base_log = np.log1p(y_base)
dtrain_base = xgb.DMatrix(X_base, label=y_base_log, missing=np.nan)

# ==========================================
# 4. TAHAP 1: PELATIHAN BASE MODEL (MAKRO SEKTOR)
# ==========================================
base_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',   # dihitung di ruang log1p, bukan skala volume asli
    'learning_rate': 0.05,
    'max_depth': 6,
    'subsample': 0.8
}

print("\n[Tahap 1] Melatih Base Model (Pengetahuan Sektor Gabungan)...")
base_model = xgb.train(base_params, dtrain_base, num_boost_round=100)

base_model_path = 'base_transport_model.json'
base_model.save_model(base_model_path)
print(f"✅ Base Model selesai dilatih!")
files.download(base_model_path)

# ==========================================
# 5. TAHAP 2: TRANSFER LEARNING UNTUK SETIAP PERUSAHAAN
# ==========================================
tune_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'learning_rate': 0.01,  # Rate kecil agar tidak merusak pengetahuan Base Model
    'max_depth': 4
}

# ---- metrik evaluasi (skala volume ASLI, setelah expm1) -----------------
# WMAPE/SMAPE dipakai selain MAE karena volume tipis (mis. BLOG.JK) bisa
# punya banyak hari dengan actual mendekati nol -- lihat testing/metrics.py
# untuk penjelasan lebih lengkap kenapa MAPE polos tidak dipakai di sini.
def _wmape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.sum(np.abs(actual))
    return float("nan") if denom == 0 else float(np.sum(np.abs(actual - pred)) / denom * 100)

def _smape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = (np.abs(actual) + np.abs(pred)) / 2.0
    safe = np.where(denom == 0, 1.0, denom)
    err = np.where(denom == 0, 0.0, np.abs(actual - pred) / safe)
    return float(np.mean(err) * 100)

target_companies = df['symbol'].unique()
print(f"\nDitemukan {len(target_companies)} perusahaan. Memulai fine-tuning untuk semuanya...")

summary_rows = []

for target_company in target_companies:
    # B. Data Fine-Tuning (Filter spesifik untuk perusahaan target)
    df_tune = df[df['symbol'] == target_company].copy()
    X_tune = df_tune[XGBOOST_FEATURES]
    y_tune = df_tune[TARGET_COLUMN]
    y_tune_log = np.log1p(y_tune)
    dtrain_tune = xgb.DMatrix(X_tune, label=y_tune_log, missing=np.nan)

    print(f"\n[Tahap 2] Melakukan Transfer Learning khusus untuk {target_company}...")
    # Injeksi transfer learning menggunakan parameter `xgb_model`
    transfer_model = xgb.train(
        tune_params,
        dtrain_tune,
        num_boost_round=50,
        xgb_model=base_model_path
    )

    finetuned_model_path = f'finetuned_{target_company.replace(".JK", "")}_model.json'
    transfer_model.save_model(finetuned_model_path)
    print(f"✅ Fine-Tuned Model selesai dilatih untuk {target_company}!")

    # ==========================================
    # 6. EVALUASI PADA SKALA VOLUME ASLI (bukan log) + UNDUH OTOMATIS
    # ==========================================
    pred_log = transfer_model.predict(dtrain_tune)
    df_tune['Predicted_Volume_T_plus_1'] = np.expm1(pred_log)  # <-- inverse transform
    df_tune['Predicted_Volume_T_plus_1'] = df_tune['Predicted_Volume_T_plus_1'].clip(lower=0)

    mae = mean_absolute_error(y_tune, df_tune['Predicted_Volume_T_plus_1'])
    wmape = _wmape(y_tune, df_tune['Predicted_Volume_T_plus_1'])
    smape = _smape(y_tune, df_tune['Predicted_Volume_T_plus_1'])
    summary_rows.append({'symbol': target_company, 'n_rows': len(df_tune),
                          'MAE': mae, 'WMAPE_%': wmape, 'SMAPE_%': smape})
    print(f"   MAE={mae:,.0f}  WMAPE={wmape:.2f}%  SMAPE={smape:.2f}%  (n={len(df_tune)})")
    print(df_tune[['Date', 'target_volume_T_plus_1', 'Predicted_Volume_T_plus_1']].head())

    print(f"📥 Mengunduh model JSON untuk {target_company} ke laptop Anda...")
    files.download(finetuned_model_path)

print("\n🎉 Semua perusahaan selesai di-fine-tune!")
print("\n=== RINGKASAN AKURASI PER SIMBOL (skala volume asli, training set) ===")
summary_df = pd.DataFrame(summary_rows).round(2)
print(summary_df.to_string(index=False))
print("\n⚠️  Catatan: angka di atas dihitung pada data TRAINING itu sendiri (tidak ada "
      "hold-out/validation split di script ini), jadi ini ukuran seberapa baik model "
      "menghafal data, bukan seberapa baik ia akan generalisasi ke hari-hari baru. "
      "Untuk angka yang bisa dipercaya, evaluasi model hasil training ini lewat "
      "testing/run_test.py dengan data yang TIDAK ikut dilatih.")
