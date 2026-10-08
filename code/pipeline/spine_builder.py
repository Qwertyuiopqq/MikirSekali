"""
spine_builder.py
-----------------
Stage 1 of the pipeline.

Reads every "Histori 5 taun terakhir*" file, stitches them into one
long DataFrame, expands each symbol onto a *daily* calendar spine (so
there are no gaps for non-trading days), forward-fills prices, and
engineers the base momentum/calendar features.

Public entry point: `build_raw_spine(paths) -> pd.DataFrame`
Produces: MCS_raw.csv

The forecast target `target_volume_T_plus_1` is the volume of the NEXT TRADING SESSION
(config.XGB_TARGET_MODE = "next_trading_day": Friday -> Monday, the day before a holiday -> the
day after it). The old definition, tomorrow's CALENDAR day, is 0 on weekends and holidays; that put
zeros into 37% of the training targets, pulled every forecast down ~2x and made Friday/Saturday
forecasts ~0. Rows without a target (the newest session and the weekend after it, or a session
more than config.TARGET_MAX_GAP_DAYS away) are KEPT: they are exactly the rows a real forecast is
needed for, and the trainer drops them itself.
"""
import glob
import os

import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:  # running as a plain script, not as a package
    import config


def _load_history_files(history_folder: str, glob_pattern: str) -> pd.DataFrame:
    """Load and concatenate all per-symbol history files into one frame."""
    all_files = glob.glob(os.path.join(history_folder, glob_pattern))
    if not all_files:
        raise FileNotFoundError(
            f"No history files found matching '{glob_pattern}' in '{history_folder}'"
        )

    # Definisikan langsung path dan template nama file sesuai infomu agar tidak error dari config
    new_data_path = "../../dataset/csv"
    new_data_tpl = "daily_transactions_{symbol}_2021_2025.csv"

    frames = []
    for file in all_files:
        filename = os.path.basename(file)
        # Ekstrak symbol dari nama file lama (misal: "Histori 5 taun terakhir ASSA.JK.csv")
        symbol = filename.split("terakhir ")[-1].replace(".csv", "").replace(".xlsx", "")
        
        # Bentuk path untuk file data baru menggunakan variabel lokal di atas
        new_file_name = new_data_tpl.format(symbol=symbol)
        new_file_path = os.path.join(new_data_path, new_file_name)
        
        if os.path.exists(new_file_path):
            # --- SKENARIO 1: DATA BARU DITEMUKAN ---
            try:
                df = pd.read_csv(new_file_path)
                
                # Sesuaikan nama kolom dengan format ekspektasi pipeline
                df = df.rename(columns={
                    "date": "Date",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume"
                })
                
                if "Dividends" not in df.columns:
                    df["Dividends"] = 0
                if "Stock Splits" not in df.columns:
                    df["Stock Splits"] = 0
                    
            except Exception as e:
                print(f"[Warning] Gagal memuat file baru untuk {symbol}: {e}. Fallback ke data lama.")
                # Fallback ke data lama
                try:
                    df = pd.read_csv(file)
                except Exception:
                    df = pd.read_excel(file)
        else:
            # --- SKENARIO 2: DATA BARU TIDAK ADA (FALLBACK KE DATA LAMA SECARA OTOMATIS) ---
            try:
                df = pd.read_csv(file)
            except Exception:
                df = pd.read_excel(file)
                
        df["symbol"] = symbol
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def _build_calendar_spine(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Cross every symbol with every calendar day between the min/max dates."""
    df_raw["Date"] = pd.to_datetime(df_raw["Date"]).dt.tz_localize(None).dt.normalize()

    full_calendar = pd.date_range(start=df_raw["Date"].min(), end=df_raw["Date"].max(), freq="D")
    unique_symbols = df_raw["symbol"].unique()

    spine_idx = pd.MultiIndex.from_product([full_calendar, unique_symbols], names=["Date", "symbol"])
    df_spine = pd.DataFrame(index=spine_idx).reset_index()

    df_master = pd.merge(df_spine, df_raw, on=["Date", "symbol"], how="left")
    return df_master.sort_values(by=["symbol", "Date"]).reset_index(drop=True)


def _fill_and_flag(df_master: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill prices, zero-fill volumes/corp-actions, add calendar flags."""
    price_columns = ["Open", "High", "Low", "Close"]
    # A raw row with Volume == 0 is a RECORD, not a trade (corporate actions, halts): its prices are not market
    # prices. TMAS 2023-05-23 is a split record priced 38.58 beside a real ~265, which made the following session
    # look like +560% and polluted daily_return_pct / volatility_7d / the moving averages for weeks. Treat such rows
    # like any non-trading day: carry the last session's prices forward.
    no_trade = df_master["Volume"].fillna(0) <= 0
    df_master.loc[no_trade, price_columns] = np.nan
    df_master[price_columns] = df_master.groupby("symbol")[price_columns].ffill()

    df_master["Volume"] = df_master["Volume"].fillna(0)
    df_master["Dividends"] = df_master["Dividends"].fillna(0)
    df_master["Stock Splits"] = df_master["Stock Splits"].fillna(0)

    df_master["day_of_week"] = df_master["Date"].dt.dayofweek
    df_master["is_weekend"] = df_master["day_of_week"].isin([5, 6]).astype(int)
    df_master["is_dividend"] = (df_master["Dividends"] > 0).astype(int)
    df_master["is_stock_split"] = (df_master["Stock Splits"] > 0).astype(int)
    return df_master


def next_session_target(df: pd.DataFrame, max_gap_days: int = None) -> pd.Series:
    """
    For every row, the Volume of the first LATER row on which the stock traded (Volume > 0),
    per symbol; NaN when there is none yet (the newest rows) or when that session is more than
    `max_gap_days` calendar days away (a suspension: "volume the day trading resumes" is not a
    next-day forecast). `df` needs symbol, Date and Volume and must be sorted by (symbol, Date).
    """
    max_gap_days = config.TARGET_MAX_GAP_DAYS if max_gap_days is None else max_gap_days
    sym = df["symbol"]
    traded = df["Volume"] > 0
    # shift(-1) = "strictly later"; bfill = "the first one that exists" (both inside each symbol)
    next_volume = df["Volume"].where(traded).groupby(sym).shift(-1).groupby(sym).bfill()
    next_date = df["Date"].where(traded).groupby(sym).shift(-1).groupby(sym).bfill()
    gap_days = (next_date - df["Date"]).dt.days
    return next_volume.where(gap_days <= max_gap_days)


def _add_momentum_features(df_master: pd.DataFrame) -> pd.DataFrame:
    """
    Per-symbol rolling/momentum features + the next-day volume target.

    Implemented with groupby().transform()/shift() rather than
    groupby().apply() so the 'symbol' column is never silently dropped
    (pandas >= 2.2 excludes the group key from the frame passed into
    apply()) and so the computation stays vectorized.
    """
    df_master = df_master.sort_values(["symbol", "Date"]).reset_index(drop=True)
    g = df_master.groupby("symbol")["Close"]

    df_master["daily_return_pct"] = g.pct_change()
    df_master["MA_7_Close"] = g.transform(lambda s: s.rolling(window=7, min_periods=1).mean())
    df_master["MA_30_Close"] = g.transform(lambda s: s.rolling(window=30, min_periods=1).mean())

    g_vol = df_master.groupby("symbol")["Volume"]
    df_master["MA_7_Volume"] = g_vol.transform(lambda s: s.rolling(window=7, min_periods=1).mean())
    if config.XGB_TARGET_MODE == config.XGB_LEGACY_TARGET_MODE:
        df_master["target_volume_T_plus_1"] = g_vol.shift(-1)      # old: tomorrow's CALENDAR day (0 on weekends)
    else:
        df_master["target_volume_T_plus_1"] = next_session_target(df_master)

    df_master["volatility_7d"] = df_master.groupby("symbol")["daily_return_pct"].transform(
        lambda s: s.rolling(window=7, min_periods=1).std()
    )
    return df_master


def build_raw_spine(paths) -> pd.DataFrame:
    """
    Build the calendar-complete raw spine (Stage 1 output -> MCS_raw.csv).

    Parameters
    ----------
    paths : mcs_pipeline.config.Paths

    Returns
    -------
    pd.DataFrame ready to be saved as MCS_raw.csv
    """
    print("[spine_builder] Membaca file histori mentah...")
    df_raw = _load_history_files(paths.history_folder, paths.history_glob_pattern)

    print("[spine_builder] Membangun kalender penuh per simbol...")
    df_master = _build_calendar_spine(df_raw)
    df_master = _fill_and_flag(df_master)

    print("[spine_builder] Menghitung fitur momentum & target...")
    df_master = _add_momentum_features(df_master)
    if config.XGB_TARGET_MODE == config.XGB_LEGACY_TARGET_MODE:
        df_master = df_master.dropna(subset=["target_volume_T_plus_1"])
    # (new mode: rows without a target stay -- they are the rows that get the real forecast)

    return df_master.reset_index(drop=True)
