import pandas as pd
import numpy as np
import xgboost as xgb
import os
import joblib # Hapus jika Anda tidak menggunakan joblib untuk save/load model

class BusinessHealthFuzzySystem:
    def __init__(self):
        self.weights_real = {
            'sentiment': 0.30,
            'trend': 0.40,
            'stability': 0.30,
            'xgboost': 0.00
        }
        self.weights_predict = {
            'sentiment': 0.20,
            'trend': 0.25,
            'stability': 0.15,
            'xgboost': 0.40
        }

    def _fuzzify_sentiment(self, sentiment_score):
        return np.clip((sentiment_score + 0.5), 0, 1)

    def _fuzzify_trend(self, return_pct, ma_7, ma_30):
        fuzzy_return = np.clip((return_pct + 0.02) / 0.05, 0, 1)
        trend_multiplier = np.where(ma_7 > ma_30, 1.0, 0.5)
        return np.clip(fuzzy_return * trend_multiplier, 0, 1)

    def _fuzzify_stability(self, volatility):
        max_acceptable_volatility = 0.05
        return np.clip(1 - (volatility / max_acceptable_volatility), 0, 1)
        
    def _fuzzify_xgboost(self, xgb_prediction, max_expected_value=1.0):
        return np.clip(xgb_prediction / max_expected_value, 0, 1)

    def calculate_health_score(self, df, mode='real', xgb_col=None):
        weights = self.weights_predict if mode == 'predict' else self.weights_real
        
        f_sent = self._fuzzify_sentiment(df['daily_news_sentiment'])
        f_trend = self._fuzzify_trend(df['daily_return_pct'], df['MA_7_Close'], df['MA_30_Close'])
        f_stab = self._fuzzify_stability(df['volatility_7d'])
        
        base_score = (
            (f_sent * weights['sentiment']) +
            (f_trend * weights['trend']) +
            (f_stab * weights['stability'])
        )
        
        if mode == 'predict' and xgb_col is not None:
            f_xgb = self._fuzzify_xgboost(df[xgb_col])
            total_score = base_score + (f_xgb * weights['xgboost'])
        else:
            total_score = base_score
            
        return np.round(total_score * 100, 2)


# ==========================================
# EKSEKUSI PROGRAM
# ==========================================
if __name__ == "__main__":
    import os
    import xgboost as xgb
    
    print("Mulai membaca dataset kalender spline... Mohon tunggu sebentar.")
    
    try:
        # 1. Membaca Master Dataset (Belum ada prediksi)
        file_path = '../XGBoost/final_master_dataset_xgboost.csv'
        df = pd.read_csv(file_path)
        
        # 2. Buat kolom kosong untuk menampung hasil prediksi
        kolom_prediksi = 'target_volume_T_plus_1_pred'
        df[kolom_prediksi] = np.nan
        
        # [PENTING] Tentukan fitur (X) yang Anda gunakan saat training XGBoost!
        # Pastikan list ini SAMA PERSIS urutannya dengan saat Anda melatih model.
        # [PENTING] Fitur (X) disesuaikan persis dengan saat model dilatih
        xgb_features = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 
            'Stock Splits', 'day_of_week', 'is_weekend', 'is_dividend', 
            'is_stock_split', 'daily_return_pct', 'MA_7_Close', 'MA_30_Close', 
            'MA_7_Volume', 'volatility_7d', 'daily_news_sentiment', 
            'daily_news_count', 'idx_macro_close'
        ]

        # 3. Looping untuk setiap perusahaan (symbol)
        perusahaan_list = df['symbol'].unique()
        print(f"\nDitemukan {len(perusahaan_list)} perusahaan. Memulai proses prediksi...")
        
        for sym in perusahaan_list:
            # Cari baris yang miliknya perusahaan ini saja
            mask = df['symbol'] == sym
            X_infer = df.loc[mask, xgb_features]
            
            # --- PERUBAHAN NAMA FILE DI SINI ---
            # Hilangkan '.JK' dari simbol (misal 'ASSA.JK' jadi 'ASSA')
            company_name = sym.replace('.JK', '')
            
            # Sesuaikan path folder-nya jika file .json tidak berada langsung di dalam folder XGBoost
            model_path = f'../XGBoost/finetuned_{company_name}_model.json' 
            
            if os.path.exists(model_path):
                # Load model JSON langsung pakai bawaan XGBoost (tanpa joblib)
                model = xgb.XGBRegressor()
                model.load_model(model_path)
                
                df.loc[mask, kolom_prediksi] = model.predict(X_infer)
                print(f"  [+] Prediksi sukses untuk {sym}")
            else:
                print(f"  [-] WARNING: File model tidak ditemukan untuk {sym} ({model_path})")

        # 4. Hapus baris yang nilai teknikal atau prediksinya masih kosong (NaN)
        df.dropna(subset=['MA_7_Close', 'MA_30_Close', 'volatility_7d', kolom_prediksi], inplace=True)

        # 5. Jalankan Sistem Fuzzy untuk REAL dan PREDIKSI
        print("\nMenghitung Skor Kesehatan Bisnis Terpadu...")
        fuzzy_sys = BusinessHealthFuzzySystem()
        
        df['health_score_real'] = fuzzy_sys.calculate_health_score(df, mode='real')
        df['health_score_predict'] = fuzzy_sys.calculate_health_score(df, mode='predict', xgb_col=kolom_prediksi)

        # 6. Tampilkan 10 Output Pertama untuk Setiap Perusahaan
        print("\n--- 10 OUTPUT PERTAMA UNTUK TIAP PERUSAHAAN ---")
        kolom_tampil = ['Date', 'symbol', 'health_score_real', 'health_score_predict']
        top_10_per_perusahaan = df[kolom_tampil].groupby('symbol').head(10)
        
        print(top_10_per_perusahaan.to_string(index=False))

        # 7. Ekspor menjadi File Tabel (CSV) Baru yang Sudah Lengkap
        output_path = '../XGBoost/dataset_master_with_predictions_and_health.csv'
        df.to_csv(output_path, index=False)
        print(f"\nSelesai! Tabel final (Master + XGBoost + Fuzzy) disimpan di:\n{output_path}")
        
    except FileNotFoundError:
        print(f"\nERROR: File dataset utama tidak ditemukan di {file_path}")
    except Exception as e:
        print(f"\nTerjadi kesalahan sistem: {e}")