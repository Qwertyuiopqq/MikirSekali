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
"""
import glob
import os

import numpy as np
import pandas as pd


def _load_history_files(history_folder: str, glob_pattern: str) -> pd.DataFrame:
    """Load and concatenate all per-symbol history files into one frame."""
    all_files = glob.glob(os.path.join(history_folder, glob_pattern))
    if not all_files:
        raise FileNotFoundError(
            f"No history files found matching '{glob_pattern}' in '{history_folder}'"
        )

    frames = []
    for file in all_files:
        filename = os.path.basename(file)
        symbol = filename.split("terakhir ")[-1].replace(".csv", "").replace(".xlsx", "")
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
    df_master[price_columns] = df_master.groupby("symbol")[price_columns].ffill()

    df_master["Volume"] = df_master["Volume"].fillna(0)
    df_master["Dividends"] = df_master["Dividends"].fillna(0)
    df_master["Stock Splits"] = df_master["Stock Splits"].fillna(0)

    df_master["day_of_week"] = df_master["Date"].dt.dayofweek
    df_master["is_weekend"] = df_master["day_of_week"].isin([5, 6]).astype(int)
    df_master["is_dividend"] = (df_master["Dividends"] > 0).astype(int)
    df_master["is_stock_split"] = (df_master["Stock Splits"] > 0).astype(int)
    return df_master


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
    df_master["target_volume_T_plus_1"] = g_vol.shift(-1)

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
    df_master = df_master.dropna(subset=["target_volume_T_plus_1"])

    return df_master.reset_index(drop=True)
