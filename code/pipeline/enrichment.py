"""
enrichment.py
-------------
Stage 2 of the pipeline.

Adds the extra signals needed by both the fuzzy health system and the
XGBoost model on top of the raw spine:
  - a mock/derived NLP sentiment score per day
  - macro index (IDX) closing price, forward-filled
  - fundamental ratios (ROE, ROA, gross margin, etc.) merged by symbol

Every step is wrapped in its own try/except so a missing optional data
source degrades gracefully instead of crashing the whole pipeline
(mirrors the original script's behaviour).

Public entry point: `enrich_features(df_master, paths) -> pd.DataFrame`
"""
import json
import os

import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:  # running as a plain script, not as a package
    import config


def _generate_mock_sentiment(row) -> float:
    """Mock sentiment: mostly noise, with a positive bump on corp actions.
    Used only as a fallback when no real sentiment file is available."""
    base_score = np.random.normal(0, 0.2)
    if row.get("is_dividend", 0) == 1 or row.get("is_stock_split", 0) == 1:
        base_score += np.random.uniform(0.5, 0.9)
    return max(min(base_score, 1.0), -1.0)


def _load_real_sentiment(sentiment_csv_path: str) -> pd.DataFrame:
    """
    Load a real sentiment file and normalize it to columns
    ['Date', 'symbol', 'daily_news_sentiment', 'daily_news_count'].

    Expected input columns (flexible naming, case-insensitive):
      - a date column ('Date' or 'date')
      - 'symbol'
      - a sentiment score column (looked for as 'daily_news_sentiment',
        else anything containing 'sentiment')
      - optionally a news-count column (looked for as 'daily_news_count',
        else anything containing 'count'); defaults to 0 if absent.

    Raises on any failure so the caller can fall back to mock data.
    """
    df_sent = pd.read_csv(sentiment_csv_path)

    date_col = "Date" if "Date" in df_sent.columns else ("date" if "date" in df_sent.columns else None)
    if date_col is None or "symbol" not in df_sent.columns:
        raise ValueError("sentiment file must have a Date column and a 'symbol' column")
    df_sent["Date"] = pd.to_datetime(df_sent[date_col]).dt.tz_localize(None).dt.normalize()

    if "daily_news_sentiment" in df_sent.columns:
        sent_col = "daily_news_sentiment"
    else:
        candidates = [c for c in df_sent.columns if "sentiment" in c.lower()]
        if not candidates:
            raise ValueError("no sentiment score column found in sentiment file")
        sent_col = candidates[0]

    if "daily_news_count" in df_sent.columns:
        count_col = "daily_news_count"
    else:
        candidates = [c for c in df_sent.columns if "count" in c.lower()]
        count_col = candidates[0] if candidates else None

    out = df_sent[["Date", "symbol", sent_col]].rename(columns={sent_col: "daily_news_sentiment"})
    out["daily_news_count"] = df_sent[count_col] if count_col else 0
    return out.drop_duplicates(subset=["Date", "symbol"], keep="last")


def _add_sentiment(df_master: pd.DataFrame, sentiment_csv_path: str, seed: int) -> pd.DataFrame:
    """
    Conditional sentiment injection:
      - If `sentiment_csv_path` exists and loads cleanly, merge the REAL
        scores in by (Date, symbol). Any rows the real file doesn't cover
        get a neutral fallback (0 sentiment, 0 count) rather than crashing.
      - Otherwise (file missing or malformed), generate the mock/derived
        sentiment score, exactly like the original script did.
    """
    if sentiment_csv_path and os.path.exists(sentiment_csv_path):
        try:
            print(f"[enrichment] Ditemukan file sentimen asli: {sentiment_csv_path} -> menggabungkan data nyata...")
            df_real_sent = _load_real_sentiment(sentiment_csv_path)
            before_cols = set(df_master.columns)
            df_master = pd.merge(df_master, df_real_sent, on=["Date", "symbol"], how="left")

            missing_mask = df_master["daily_news_sentiment"].isna()
            n_missing = int(missing_mask.sum())
            if n_missing:
                print(f"   -> {n_missing} baris tidak ada di file sentimen, diisi skor netral (0).")
                df_master.loc[missing_mask, "daily_news_sentiment"] = 0.0
                df_master.loc[missing_mask, "daily_news_count"] = 0

            print(f"   ✅ Sentimen asli berhasil digabungkan ({len(df_real_sent)} baris sumber).")
            return df_master
        except Exception as e:
            print(f"   ⚠️ Gagal membaca file sentimen asli ({e}), fallback ke mock sentiment...")

    print("[enrichment] Tidak ada file sentimen asli -> menyuntikkan mock sentiment...")
    np.random.seed(seed)
    df_master = df_master.copy()
    df_master["daily_news_sentiment"] = df_master.apply(_generate_mock_sentiment, axis=1)
    df_master["daily_news_count"] = np.random.randint(0, 15, size=len(df_master))
    return df_master


def _add_macro(df_master: pd.DataFrame, idx_csv_path: str) -> pd.DataFrame:
    print("[enrichment] Menyuntikkan data makro (IDX)...")
    try:
        df_idx = pd.read_csv(idx_csv_path)
        date_col = "date" if "date" in df_idx.columns else ("Date" if "Date" in df_idx.columns else df_idx.columns[0])
        df_idx["Date"] = pd.to_datetime(df_idx[date_col]).dt.tz_localize(None).dt.normalize()

        close_col = "Close"
        if "Close" not in df_idx.columns:
            possible_cols = [c for c in df_idx.columns if "close" in c.lower() or "last" in c.lower()]
            close_col = possible_cols[0] if possible_cols else df_idx.columns[1]

        print(f"   -> Menggunakan kolom '{close_col}' sebagai data makro.")
        df_idx = df_idx[["Date", close_col]].rename(columns={close_col: "idx_macro_close"})
        df_master = pd.merge(df_master, df_idx, on="Date", how="left")
        df_master["idx_macro_close"] = df_master["idx_macro_close"].ffill()
    except Exception as e:
        print(f"   ⚠️ Melewati injeksi makro: {e}")
    return df_master


def _add_fundamentals(df_master: pd.DataFrame, fundamental_json_path: str) -> pd.DataFrame:
    print("[enrichment] Menyuntikkan metrik fundamental...")
    try:
        with open(fundamental_json_path) as f:
            company_report = json.load(f)

        if "financial_ratios" in company_report:
            df_ratios = pd.DataFrame(company_report["financial_ratios"])
        else:
            df_ratios = pd.concat(
                [pd.DataFrame(records) for key, records in company_report.items()
                 if key == "financial_ratios" or isinstance(records, list)]
            )

        if not df_ratios.empty and "symbol" in df_ratios.columns:
            latest_ratios = df_ratios.drop_duplicates(subset=["symbol"], keep="last")
            keep_cols = ["symbol"] + [c for c in latest_ratios.columns
                                       if c in ["ROE", "ROA", "gross_margin", "debt_to_equity", "forward_pe"]]
            latest_ratios = latest_ratios[keep_cols]
            df_master = pd.merge(df_master, latest_ratios, on="symbol", how="left")
    except Exception as e:
        print(f"   ⚠️ Melewati injeksi fundamental (struktur JSON belum sesuai mapping): {e}")
    return df_master


def enrich_features(df_master: pd.DataFrame, paths) -> pd.DataFrame:
    """
    Add sentiment, macro, and fundamental features on top of the raw spine.

    Parameters
    ----------
    df_master : pd.DataFrame  (output of spine_builder.build_raw_spine)
    paths : mcs_pipeline.config.Paths

    Returns
    -------
    pd.DataFrame, enriched, ready for either fuzzy scoring or XGBoost inference
    """
    df_master = df_master.copy()
    df_master = _add_sentiment(df_master, paths.sentiment_scores_csv, seed=config.RANDOM_SEED)
    df_master = _add_macro(df_master, paths.idx_market_summary_csv)
    df_master = _add_fundamentals(df_master, paths.fundamental_json)
    return df_master
