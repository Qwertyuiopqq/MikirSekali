import pandas as pd
import glob
import os
import numpy as np

# 1. KONFIGURASI DIREKTORI
# Sesuaikan path ini dengan folder di VSCode Anda ("image_b703a7.png")
folder_path = r'../../dataset/history5y'
# Karena file bisa berupa .csv atau .xlsx (Excel), kita gunakan pola wildcard
all_files = glob.glob(os.path.join(folder_path, "Histori 5 taun terakhir*"))

list_df = []

# ==========================================
# 2. DATA INGESTION & EKSTRAKSI SYMBOL
# ==========================================
for file in all_files:
    # Mengekstrak nama emiten dari nama file (misal: "Histori 5 taun terakhir ASSA.JK.csv" -> "ASSA.JK")
    filename = os.path.basename(file)
    symbol = filename.split("terakhir ")[-1].replace(".csv", "").replace(".xlsx", "")
    
    # Membaca file (Gunakan read_excel jika format aslinya benar-benar xlsx)
    try:
        df = pd.read_csv(file)
    except:
        df = pd.read_excel(file)
        
    df['symbol'] = symbol
    list_df.append(df)

# Gabungkan seluruh data 10 perusahaan menjadi satu tabel rata
df_raw = pd.concat(list_df, ignore_index=True)

# Normalisasi zona waktu kolom Date (hapus +07:00 agar rapi)
df_raw['Date'] = pd.to_datetime(df_raw['Date']).dt.tz_localize(None).dt.normalize()

# ==========================================
# 3. MEMBANGUN KERANGKA MASTER CALENDAR SPINE
# ==========================================
# Mencari tanggal paling awal dan paling akhir dari seluruh dataset
min_date = df_raw['Date'].min()
max_date = df_raw['Date'].max()
unique_symbols = df_raw['symbol'].unique()

# Membuat kalender absolut (setiap hari tanpa jeda, termasuk Sabtu & Minggu)
full_calendar = pd.date_range(start=min_date, end=max_date, freq='D')

# Melakukan Cross-Join (Cartesian Product) antara semua tanggal dan 10 emiten
spine_idx = pd.MultiIndex.from_product(
    [full_calendar, unique_symbols], 
    names=['Date', 'symbol']
)
df_spine = pd.DataFrame(index=spine_idx).reset_index()

# Menyatukan data mentah ke kerangka kalender (Left Join)
df_master = pd.merge(df_spine, df_raw, on=['Date', 'symbol'], how='left')

# ==========================================
# 4. PENANGANAN NILAI KOSONG (IMPUTASI AKHIR PEKAN)
# ==========================================
# Urutkan berdasarkan perusahaan dan tanggal
df_master = df_master.sort_values(by=['symbol', 'Date']).reset_index(drop=True)

# Harga saham di akhir pekan sama dengan harga penutupan hari Jumat (Forward-Fill)
price_columns = ['Open', 'High', 'Low', 'Close']
df_master[price_columns] = df_master.groupby('symbol')[price_columns].ffill()

# Volume, Dividen, dan Stock Split pada hari libur adalah 0
df_master['Volume'] = df_master['Volume'].fillna(0)
df_master['Dividends'] = df_master['Dividends'].fillna(0)
df_master['Stock Splits'] = df_master['Stock Splits'].fillna(0)

# ==========================================
# 5. FEATURE ENGINEERING UNTUK XGBOOST & FUZZY
# ==========================================
# A. Ekstraksi Fitur Musiman (Time-Based)
df_master['day_of_week'] = df_master['Date'].dt.dayofweek
df_master['is_weekend'] = df_master['day_of_week'].isin([5, 6]).astype(int)

# B. Ekstraksi Katalis (Event Encoding - Dummy Variables)
df_master['is_dividend'] = (df_master['Dividends'] > 0).astype(int)
df_master['is_stock_split'] = (df_master['Stock Splits'] > 0).astype(int)

# C. Ekstraksi Momentum & Tren (Lag, Rolling, & Volatility)
# Kita hitung fitur ini per kelompok simbol (per perusahaan)
def generate_momentum_features(group):
    # Persentase perubahan harian (Sensitivitas Harga)
    group['daily_return_pct'] = group['Close'].pct_change()
    
    # Moving Average 7 Hari & 30 Hari (Momentum Jangka Pendek & Menengah)
    group['MA_7_Close'] = group['Close'].rolling(window=7, min_periods=1).mean()
    group['MA_30_Close'] = group['Close'].rolling(window=30, min_periods=1).mean()
    group['MA_7_Volume'] = group['Volume'].rolling(window=7, min_periods=1).mean()
    
    # Volatilitas Harga 7 Hari (Standard Deviation)
    group['volatility_7d'] = group['daily_return_pct'].rolling(window=7, min_periods=1).std()
    
    # Target Variabel (Y) untuk XGBoost Forecasting: Memprediksi Volume Esok Hari (T+1)
    group['target_volume_T_plus_1'] = group['Volume'].shift(-1)
    
    return group

df_master = df_master.groupby('symbol', group_keys=False).apply(generate_momentum_features)

# Hapus baris terakhir per perusahaan yang nilai target_volume_T_plus_1 nya pasti NaN
df_master = df_master.dropna(subset=['target_volume_T_plus_1'])

# Simpan Master Calendar Spine ke CSV
output_filename = 'master_calendar_spine_mikro.csv'
df_master.to_csv(output_filename, index=False)
print(f"✅ Master Calendar Spine berhasil dibuat! Tersimpan sebagai: {output_filename}")
print(df_master[['Date', 'symbol', 'Close', 'MA_7_Close', 'is_dividend', 'target_volume_T_plus_1']].head(10))