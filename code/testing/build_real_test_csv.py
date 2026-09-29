"""
build_real_test_csv.py
------------------------
Turns the raw "real data" export (daily_transaction_for_10_top_industy_
in_transportation.csv + IDX_market_summary.csv + the fundamental /
corporate-action JSONs) into a single CSV in the exact MCS_features
"master calendar spine" shape that testing/run_test.py expects --
for EVERY company present in the transaction file, over the date
range actually present in that file (no invented dates).

It reuses the pipeline's own spine-building and enrichment logic
(pipeline.spine_builder, pipeline.enrichment) instead of
re-implementing the math, so the output is guaranteed to match what
`pipeline.py prepare` would have produced from this data.

What it does
------------
1. Reads the raw daily transaction file (symbol, date, open/high/low/
   close, volume) and renames it to the pipeline's internal schema
   (Date, Open, High, Low, Close, Volume).
2. Adds real Dividends / Stock Splits flags by matching each symbol's
   ex_date / split date from the corporate-action JSON against the
   dates actually present in the transaction file (falls back to 0 if
   no corporate-action file is given).
3. Builds the full daily calendar spine per symbol (fills weekends /
   holidays by forward-filling prices, zero-filling volume) and the
   same momentum/target features as pipeline.spine_builder --
   including the real `target_volume_T_plus_1` ground truth, since
   this is historical data.
4. Enriches with macro (IDX_market_summary.csv) and fundamentals
   (company_report.json), and mock sentiment (no real sentiment file
   was provided for this dataset), via pipeline.enrichment.
5. Writes the result as one CSV covering every symbol in the input.

Usage
-----
    python build_real_test_csv.py \\
        --transactions "New folder/daily_transaction_for_10_top_industy_in_transportation.csv" \\
        --idx-market-summary "New folder/IDX_market_summary.csv" \\
        --fundamental-json "New folder/top10-transportation-by-marketcap-company_report.json" \\
        --corporate-action-json "New folder/top10-transportation-by-market-cap-corporate-action.json" \\
        --output sample_data/mock_test_real_data.csv

Then feed the result straight into the tester:

    python run_test.py --input sample_data/mock_test_real_data.csv \\
        --xgboost-models-folder ../models/XGBoost
"""
import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from pipeline import config
from pipeline.spine_builder import _build_calendar_spine, _fill_and_flag, _add_momentum_features
from pipeline.enrichment import enrich_features


def _load_raw_transactions(transactions_csv: str) -> pd.DataFrame:
    """
    Normalize the raw daily_transaction_for_10_top_industy_in_transportation.csv
    (symbol, date, open, high, low, close, volume, market_cap) into the
    column names spine_builder expects (Date, Open, High, Low, Close, Volume).
    """
    df = pd.read_csv(transactions_csv)

    rename_map = {}
    for col in df.columns:
        low = col.lower()
        if low == "date":
            rename_map[col] = "Date"
        elif low == "open":
            rename_map[col] = "Open"
        elif low == "high":
            rename_map[col] = "High"
        elif low == "low":
            rename_map[col] = "Low"
        elif low == "close":
            rename_map[col] = "Close"
        elif low == "volume":
            rename_map[col] = "Volume"
    df = df.rename(columns=rename_map)

    required = ["Date", "symbol", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Transaction file is missing expected columns: {missing}")

    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    return df


def _add_real_corporate_actions(df: pd.DataFrame, corporate_action_json: str) -> pd.DataFrame:
    """
    Sets Dividends / Stock Splits from real ex-dividend / stock-split
    dates in the corporate-action JSON, matched by (symbol, Date).
    Falls back to all-zero columns if no file is given or it can't be
    read -- exactly like the rest of the pipeline degrades gracefully
    on optional inputs.
    """
    df = df.copy()
    df["Dividends"] = 0.0
    df["Stock Splits"] = 0.0

    if not corporate_action_json or not os.path.exists(corporate_action_json):
        print("[build_real_test_csv] Tidak ada file corporate action -> Dividends/Stock Splits diisi 0.")
        return df

    try:
        with open(corporate_action_json) as f:
            actions = json.load(f)

        if actions.get("dividend"):
            df_div = pd.DataFrame(actions["dividend"])
            df_div["Date"] = pd.to_datetime(df_div["ex_date"], errors="coerce").dt.tz_localize(None).dt.normalize()
            df_div = df_div.dropna(subset=["Date"])
            div_amt = (
                df_div.groupby(["symbol", "Date"])["dividend_amount"]
                .sum().rename("Dividends").reset_index()
            )
            df = df.drop(columns=["Dividends"]).merge(div_amt, on=["symbol", "Date"], how="left")
            df["Dividends"] = df["Dividends"].fillna(0.0)

        if actions.get("stock_split"):
            df_ss = pd.DataFrame(actions["stock_split"])
            date_col = "date" if "date" in df_ss.columns else "ex_date"
            df_ss["Date"] = pd.to_datetime(df_ss[date_col], errors="coerce").dt.tz_localize(None).dt.normalize()
            df_ss = df_ss.dropna(subset=["Date"])
            ratio_col = "split_ratio" if "split_ratio" in df_ss.columns else df_ss.columns[-1]
            ss_amt = (
                df_ss.groupby(["symbol", "Date"])[ratio_col]
                .sum().rename("Stock Splits").reset_index()
            )
            df = df.drop(columns=["Stock Splits"]).merge(ss_amt, on=["symbol", "Date"], how="left")
            df["Stock Splits"] = df["Stock Splits"].fillna(0.0)

        n_div_days = int((df["Dividends"] > 0).sum())
        n_ss_days = int((df["Stock Splits"] > 0).sum())
        print(f"[build_real_test_csv] Corporate actions asli dicocokkan: "
              f"{n_div_days} baris dividen, {n_ss_days} baris stock split.")
    except Exception as e:
        print(f"[build_real_test_csv] ⚠️ Gagal memuat corporate action ({e}), fallback ke 0.")
        df["Dividends"] = 0.0
        df["Stock Splits"] = 0.0

    return df


def build_real_test_csv(transactions_csv: str, idx_market_summary_csv: str = None,
                         fundamental_json: str = None, corporate_action_json: str = None,
                         sentiment_scores_csv: str = None, output_csv: str = None) -> pd.DataFrame:
    print(f"[build_real_test_csv] Membaca transaksi mentah: {transactions_csv}")
    df_raw = _load_raw_transactions(transactions_csv)
    symbols = sorted(df_raw["symbol"].unique())
    print(f"   -> {len(symbols)} perusahaan ditemukan: {symbols}")
    print(f"   -> Rentang tanggal pada data: {df_raw['Date'].min().date()} s/d {df_raw['Date'].max().date()}")

    df_raw = _add_real_corporate_actions(df_raw, corporate_action_json)

    print("[build_real_test_csv] Membangun kalender penuh per simbol (semua perusahaan, semua tanggal pada data)...")
    df_master = _build_calendar_spine(df_raw)
    df_master = _fill_and_flag(df_master)

    print("[build_real_test_csv] Menghitung fitur momentum & target (target_volume_T_plus_1)...")
    df_master = _add_momentum_features(df_master)
    df_master = df_master.dropna(subset=["target_volume_T_plus_1"])
    df_master = df_master.reset_index(drop=True)

    # Reuse a Paths object purely to point enrichment at the real
    # macro/fundamental files supplied on the command line; the
    # sentiment file is optional and falls back to mock sentiment
    # automatically if it's missing.
    paths = config.Paths(
        idx_market_summary_csv=idx_market_summary_csv or config.Paths().idx_market_summary_csv,
        fundamental_json=fundamental_json or config.Paths().fundamental_json,
        sentiment_scores_csv=sentiment_scores_csv or "",
    )

    print("[build_real_test_csv] Enrichment (sentiment/macro/fundamental)...")
    df_features = enrich_features(df_master, paths)

    print(f"[build_real_test_csv] Selesai: {len(df_features)} baris, "
          f"{df_features['symbol'].nunique()} perusahaan, "
          f"{df_features['Date'].min().date()} s/d {df_features['Date'].max().date()}")

    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)) or ".", exist_ok=True)
        df_features.to_csv(output_csv, index=False)
        print(f"✅ Disimpan di: {output_csv}")

    return df_features


def main():
    parser = argparse.ArgumentParser(
        description="Build an MCS_features-format test CSV (all companies, real dates) from raw real data."
    )
    parser.add_argument("--transactions", required=True,
                         help="Path to daily_transaction_for_10_top_industy_in_transportation.csv")
    parser.add_argument("--idx-market-summary", default=None, help="Path to IDX_market_summary.csv")
    parser.add_argument("--fundamental-json", default=None,
                         help="Path to top10-transportation-by-marketcap-company_report.json")
    parser.add_argument("--corporate-action-json", default=None,
                         help="Path to top10-transportation-by-market-cap-corporate-action.json "
                              "(used for real Dividends/Stock Splits flags; optional)")
    parser.add_argument("--sentiment-scores-csv", default=None,
                         help="Optional real sentiment CSV; omit to use mock sentiment.")
    parser.add_argument("--output", required=True, help="Where to write the resulting test CSV.")
    args = parser.parse_args()

    build_real_test_csv(
        transactions_csv=args.transactions,
        idx_market_summary_csv=args.idx_market_summary,
        fundamental_json=args.fundamental_json,
        corporate_action_json=args.corporate_action_json,
        sentiment_scores_csv=args.sentiment_scores_csv,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
