import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 1. Memuat variabel dari file .env
load_dotenv()
api_key = os.getenv("SECTORS_API_KEY")

if not api_key:
    raise ValueError("⚠️ API Key tidak ditemukan! Pastikan file .env sudah dikonfigurasi.")

# 2. Konfigurasi Endpoint dan Header
symbol = "ASSA.JK"
url = f"https://api.sectors.app/v2/daily/{symbol}/"
headers = {"Authorization": api_key}
csv_filename = f"daily_transaction_{symbol}_2021_2025.csv"

# 3. Baca data CSV yang sudah ada
if os.path.exists(csv_filename):
    df_existing = pd.read_csv(csv_filename)
    df_existing['date'] = pd.to_datetime(df_existing['date'])
    print(f"✅ Memuat file {csv_filename} ({len(df_existing)} baris).")
else:
    raise FileNotFoundError(f"❌ File {csv_filename} tidak ditemukan. Skrip perbaikan butuh file hasil sebelumnya.")

# 4. Konfigurasi Tanggal Penarikan
start_date = datetime.strptime("2021-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2025-12-31", "%Y-%m-%d")

current_start = start_date
new_transactions = []

print(f"🔍 Mencari rentang tanggal yang gagal ditarik...")

# 5. Looping Pengecekan dan Penarikan Ulang
while current_start <= end_date:
    current_end = current_start + timedelta(days=29)
    if current_end > end_date:
        current_end = end_date
        
    str_start = current_start.strftime("%Y-%m-%d")
    str_end = current_end.strftime("%Y-%m-%d")
    
    # Cek apakah rentang tanggal ini kosong di CSV yang ada
    mask = (df_existing['date'] >= current_start) & (df_existing['date'] <= current_end)
    
    if df_existing[mask].empty:
        print(f"🔄 Menarik ulang data rentang: {str_start} s/d {str_end}...")
        params = {"start": str_start, "end": str_end}
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                new_transactions.extend(data)
                print(f"  ✅ Sukses ditarik ({len(data)} baris)")
            else:
                print(f"  ℹ️ Sukses ditarik, tapi tidak ada transaksi di rentang ini.")
                
        except requests.exceptions.HTTPError as err:
            print(f"  ❌ Error saat menarik data ({str_start} - {str_end}): {err}")
        
        # Jeda diperpanjang menjadi 3 detik khusus untuk retry agar terhindar dari limit
        time.sleep(3.0)

    current_start = current_end + timedelta(days=1)

# 6. Gabungkan, Bersihkan Duplikat, dan Simpan
if new_transactions:
    df_new = pd.DataFrame(new_transactions)
    df_new['date'] = pd.to_datetime(df_new['date'])
    
    # Gabungkan data lama dengan data hasil perbaikan
    df_combined = pd.concat([df_existing, df_new])
    
    # Urutkan ulang berdasarkan tanggal dan buang kemungkinan duplikat
    df_combined = df_combined.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
    
    # Timpa CSV dengan data yang sudah utuh
    df_combined.to_csv(csv_filename, index=False)
    print(f"\n🎉 Perbaikan Selesai! {len(df_new)} baris baru ditambahkan.")
    print(f"📂 Total keseluruhan: {len(df_combined)} baris data tersimpan di '{csv_filename}'")
else:
    print("\n✅ Tidak ada rentang data yang kosong. CSV sudah lengkap.")