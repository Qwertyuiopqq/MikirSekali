import pandas as pd
import numpy as np
import json
import os

# ==========================================
# 1. LOAD MASTER CALENDAR SPINE (HASIL SEBELUMNYA)
# ==========================================
print("Membaca Master Calendar Spine Mikro...")
df_master = pd.read_csv('master_calendar_spine_mikro.csv')
df_master['Date'] = pd.to_datetime(df_master['Date'])

# ==========================================
# 2. INJEKSI MOCKUP SENTIMEN (IndoBERT/FinBERT)
# ==========================================
print("Menyuntikkan Mockup Skor Sentimen NLP...")
# Membuat sentimen logis: Jika hari itu harga naik (return positif) atau ada Dividen, sentimen cenderung positif
np.random.seed(42) # Agar hasil bisa direproduksi

def generate_smart_sentiment(row):
    # Skor dasar netral dengan sedikit noise
    base_score = np.random.normal(0, 0.2) 
    
    # Jika hari itu ada dividen/stock split, berita pasti ramai dan positif
    if row.get('is_dividend', 0) == 1 or row.get('is_stock_split', 0) == 1:
        base_score += np.random.uniform(0.5, 0.9)
        
    # Memastikan skor mentok di rentang [-1, 1]
    return max(min(base_score, 1.0), -1.0)

# Apply sentimen dan buat volume berita (viralitas)
df_master['daily_news_sentiment'] = df_master.apply(generate_smart_sentiment, axis=1)
df_master['daily_news_count'] = np.random.randint(0, 15, size=len(df_master)) # 0-15 berita per hari

# ==========================================
# 3. INJEKSI DATA MAKRO SEKTOR (IDX Market Summary)
# ==========================================
print("Menyuntikkan Data Makro (IDX)...")
try:
    df_idx = pd.read_csv('../../dataset/csv/IDX_market_summary.csv')
    print(f"   -> Info: Kolom yang ditemukan di file IDX: {df_idx.columns.tolist()}")
    
    # 1. Deteksi otomatis kolom Tanggal
    date_col = 'date' if 'date' in df_idx.columns else ('Date' if 'Date' in df_idx.columns else df_idx.columns[0])
    df_idx['Date'] = pd.to_datetime(df_idx[date_col]).dt.tz_localize(None).dt.normalize()
    
    # 2. Deteksi otomatis kolom harga penutupan (Close/close/Last)
    close_col = 'Close'
    if 'Close' not in df_idx.columns:
        # Cari kolom yang mengandung kata 'close' (huruf kecil/besar) atau 'last'
        possible_cols = [c for c in df_idx.columns if 'close' in c.lower() or 'last' in c.lower()]
        close_col = possible_cols[0] if possible_cols else df_idx.columns[1] # Ambil kolom ke-2 jika tidak ketemu
        
    print(f"   -> Info: Menggunakan kolom '{close_col}' sebagai data Makro.")
    
    # Ambil kolom yang dibutuhkan saja
    df_idx = df_idx[['Date', close_col]].rename(columns={close_col: 'idx_macro_close'})
    
    # Gabungkan (Left Join)
    df_master = pd.merge(df_master, df_idx, on='Date', how='left')
    df_master['idx_macro_close'] = df_master['idx_macro_close'].ffill()
    
except Exception as e:
    print(f"⚠️ Melewati injeksi Makro: {e}")

# ==========================================
# 4. INJEKSI RASIO FUNDAMENTAL (Company Report JSON)
# ==========================================
print("Menyuntikkan Metrik Fundamental (Fuzzy Inputs)...")
try:
    with open('../../dataset/json/top10-transportation-by-marketcap-company_report.json') as f:
        company_report = json.load(f)
    
    # Ekstraksi financial ratios (ROE, ROA, Debt-to-Equity)
    ratios_list = []
    # Mengiterasi JSON (asumsi berstruktur: company_report['financial_ratios'] = list of dicts)
    # Sesuaikan ekstraksi ini dengan struktur bersarang JSON API Sectors Anda
    if 'financial_ratios' in company_report:
        df_ratios = pd.DataFrame(company_report['financial_ratios'])
    else:
        # Jika JSON berisi dictionary per simbol, kita ratakan
        df_ratios = pd.concat([pd.DataFrame(records) for key, records in company_report.items() if key == 'financial_ratios' or isinstance(records, list)])

    # Jika df_ratios punya kolom 'symbol', 'ROE', 'Debt_to_Equity', dll
    # Kita gabungkan ke df_master berdasarkan 'symbol'
    if not df_ratios.empty and 'symbol' in df_ratios.columns:
        # Mengambil rasio terbaru per perusahaan untuk kesederhanaan hackathon
        latest_ratios = df_ratios.drop_duplicates(subset=['symbol'], keep='last')
        
        # Kolom rasio fundamental sangat dibutuhkan untuk Fuzzy System
        cols_to_merge = ['symbol'] + [c for c in latest_ratios.columns if c in ['ROE', 'ROA', 'gross_margin', 'debt_to_equity', 'forward_pe']]
        latest_ratios = latest_ratios[cols_to_merge]
        
        df_master = pd.merge(df_master, latest_ratios, on='symbol', how='left')
except Exception as e:
    print(f"⚠️ Melewati injeksi Fundamental (Struktur JSON belum sesuai mapping): {e}")

# ==========================================
# 5. PENYIMPANAN FINAL DATASET XGBOOST & FUZZY
# ==========================================
# Buang baris yang target volumenya masih kosong
df_master = df_master.dropna(subset=['target_volume_T_plus_1'])

output_filename = 'final_master_dataset_xgboost.csv'
df_master.to_csv(output_filename, index=False)
print(f"✅ BINGO! Final Master Dataset berhasil dibuat: {output_filename}")

# Tampilkan data dengan aman (hanya menampilkan kolom yang benar-benar ada)
cols_to_print = ['Date', 'symbol', 'MA_7_Close', 'daily_news_sentiment', 'target_volume_T_plus_1']
if 'idx_macro_close' in df_master.columns:
    cols_to_print.insert(4, 'idx_macro_close')

print(df_master[cols_to_print].head())