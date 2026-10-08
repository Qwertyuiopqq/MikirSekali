import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Memuat API key dari .env
load_dotenv()
api_key = os.getenv("SECTORS_API_KEY")

if not api_key:
    raise ValueError("⚠️ API Key tidak ditemukan! Pastikan file .env sudah dikonfigurasi.")

symbol = "WBSA.JK"
url = f"https://api.sectors.app/v2/daily/{symbol}/"
headers = {"Authorization": api_key}

start_date = datetime.strptime("2021-01-01", "%Y-%m-%d")
end_date = datetime.strptime("2025-12-31", "%Y-%m-%d")

all_transactions = []
current_start = start_date

print(f"Memulai penarikan data transaksi untuk {symbol} dengan Anti-Rate Limit...")

while current_start <= end_date:
    current_end = current_start + timedelta(days=29)
    if current_end > end_date:
        current_end = end_date
        
    str_start = current_start.strftime("%Y-%m-%d")
    str_end = current_end.strftime("%Y-%m-%d")
    
    params = {"start": str_start, "end": str_end}
    
    # Konfigurasi Retry untuk menangani Error 429
    max_retries = 5
    backoff_base = 2  # Waktu tunggu akan melipat ganda (2s, 4s, 8s, 16s, 32s)
    success = False
    
    for attempt in range(max_retries):
        try:
            print(f"🔄 Menarik rentang: {str_start} s/d {str_end} (Percobaan ke-{attempt+1})...")
            response = requests.get(url, headers=headers, params=params)
            
            # Jika terkena Too Many Requests (429)
            if response.status_code == 429:
                sleep_time = backoff_base ** (attempt + 1)
                print(f"⚠️ Terkena Rate Limit (429). Menunggu {sleep_time} detik sebelum mencoba ulang...")
                time.sleep(sleep_time)
                continue
                
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, list):
                all_transactions.extend(data)
                
            success = True
            break # Berhasil, keluar dari loop retry
            
        except requests.exceptions.HTTPError as err:
            print(f"❌ Error HTTP ({str_start} - {str_end}): {err}")
            break
        except Exception as err:
            print(f"❌ Error tak terduga: {err}")
            break
            
    if not success:
        print(f"❌ Gagal total menarik data untuk rentang {str_start} s/d {str_end}.")
        
    # Jeda aman antar rentang tanggal reguler (perbesar sedikit menjadi 2 detik)
    time.sleep(2.0)
    current_start = current_end + timedelta(days=1)

# Simpan ke CSV jika data terkumpul
if all_transactions:
    df = pd.DataFrame(all_transactions)
    df['date'] = pd.to_datetime(df['date'])
    # Mengurutkan dan menghapus duplikat data jaga-jaga ada tumpang tindih
    df = df.sort_values('date').drop_duplicates().reset_index(drop=True)
    
    # 1. Tentukan path folder tujuan
    output_dir = "../../dataset/csv"
    
    # 2. Buat foldernya secara otomatis jika belum ada (mencegah FileNotFoundError)
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. Tentukan nama file (saya sesuaikan menjadi "transactions" pakai 's' sesuai format kamu sebelumnya)
    csv_filename = f"daily_transactions_{symbol}_2021_2025.csv"
    
    # 4. Gabungkan path folder dan nama file
    full_path = os.path.join(output_dir, csv_filename)
    
    # 5. Simpan DataFrame ke full_path
    df.to_csv(full_path, index=False)
    
    print(f"\n✅ Selesai! Total {len(df)} baris data berhasil disimpan ke '{full_path}'")
else:
    print("\n⚠️ Tidak ada data yang berhasil ditarik.")