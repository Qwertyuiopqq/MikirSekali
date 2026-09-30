"""
config.py
---------
Single source of truth for every file path, column list, and tunable
constant used across the pipeline. Edit this file (or override via CLI
flags in pipeline.py) instead of hunting through each module.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Paths:
    # ---- inputs -----------------------------------------------------
    history_folder: str = "../../dataset/history5y"
    history_glob_pattern: str = "Histori 5 taun terakhir*"
    idx_market_summary_csv: str = "../../dataset/csv/IDX_market_summary.csv"
    fundamental_json: str = "../../dataset/json/top10-transportation-by-marketcap-company_report.json"
    # Optional REAL sentiment file. If it exists, enrichment.py merges it in
    # by (Date, symbol). If it's missing (or fails to load), a mock/derived
    # sentiment score is generated instead so the pipeline never blocks.
    sentiment_scores_csv: str = "../../dataset/csv/news_sentiment.csv"
    xgboost_models_folder: str = "../../models/XGBoost"
    xgboost_model_filename_tpl: str = "finetuned_{company}_model.json"
    # Base (pooled) XGBoost model, same folder as the fine-tuned ones. Only
    # used as a FALLBACK when a symbol has no fine-tuned model of its own.
    xgboost_base_model_filename: str = "base_transport_model.json"
    # Optional: LSTM-with-Hurst breakout-probability models, ONE PER COMPANY.
    # Expected layout (this is exactly what the Colab export zip unpacks to):
    #
    #   <lstm_hurst_model_path>/
    #       ASSA/  best_lstm_hurst_model.pth
    #              lstm_hurst_scaler.joblib
    #              lstm_hurst_best_params.json
    #       BIRD/  ...
    #       (one sub-folder per symbol, name = symbol without ".JK")
    #
    # A company whose sub-folder (or any of its 3 files) is missing is
    # skipped gracefully -- its breakout_probability stays NaN -- instead
    # of failing the whole pipeline (same pattern as the optional
    # sentiment file in enrichment.py).
    lstm_hurst_model_path: str = "../../models/LSTMwithHurst"
    # Fine-tuned FinBERT (HuggingFace format: config.json, tokenizer files, weights).
    # Only used by website/sentiment_trigger.py (demo sentiment badge), not by the
    # pipeline itself. Relative to pipeline/, like every other path above.
    finbert_model_folder: str = "../../models/finBERT/model_finetuned"

    # ---- outputs ------------------------------------------------------
    output_folder: str = "../../data/output"
    mcs_raw_csv: str = "MCS_raw.csv"          # Stage 1: raw calendar spine
    mcs_features_csv: str = "MCS_features.csv"  # Stage 2: raw + enrichment, ready for XGBoost training/inference
    mcs_report_csv: str = "MCS_report.csv"    # Stage 3: features + health_score_real
    mcs_predict_csv: str = "MCS_predict.csv"  # Stage 5: features + XGBoost prediction
    mcs_health_csv: str = "MCS_health.csv"    # Stage 6: predict + health_score_real + health_score_predict

    def output_path(self, filename: str) -> str:
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        return str(Path(self.output_folder) / filename)


# Feature columns fed into the XGBoost volume model.
# NOTE: order/content must exactly match what the model was trained on.
XGBOOST_FEATURES: List[str] = [
    "Open", "High", "Low", "Close", "Volume", "Dividends",
    "Stock Splits", "day_of_week", "is_weekend", "is_dividend",
    "is_stock_split", "daily_return_pct", "MA_7_Close", "MA_30_Close",
    "MA_7_Volume", "volatility_7d", "daily_news_sentiment",
    "daily_news_count", "idx_macro_close",
]

# Fuzzy health-score weights.
FUZZY_WEIGHTS_REAL = {
    "sentiment": 0.30,
    "trend": 0.40,
    "stability": 0.30,
    "xgboost": 0.00,
}
FUZZY_WEIGHTS_PREDICT = {
    "sentiment": 0.20,
    "trend": 0.25,
    "stability": 0.15,
    "xgboost": 0.40,
}
# Same as FUZZY_WEIGHTS_PREDICT but with the optional LSTM-Hurst breakout
# probability folded in. Weights below are ONE reasonable rebalancing
# (still sums to 1.0) -- not derived from anything, just kept
# proportionally similar to FUZZY_WEIGHTS_PREDICT. Change freely.
FUZZY_WEIGHTS_PREDICT_BREAKOUT = {
    "sentiment": 0.15,
    "trend": 0.20,
    "stability": 0.15,
    "xgboost": 0.30,
    "breakout": 0.20,
}

# Per-company LSTM-Hurst artifact filenames (inside <lstm_hurst_model_path>/<COMPANY>/).
LSTM_MODEL_FILENAME = "best_lstm_hurst_model.pth"
LSTM_SCALER_FILENAME = "lstm_hurst_scaler.joblib"
LSTM_PARAMS_FILENAME = "lstm_hurst_best_params.json"

# Support / resistance lines (pipeline/support_resistance.py).
SR_PIVOT_WINDOW = 5          # bars on each side needed to confirm a swing high/low
SR_LOOKBACK_TRADING_DAYS = 252   # ~1 trading year of pivots are considered
SR_CLUSTER_ATR_MULT = 0.6    # pivots closer than this many ATRs are merged into one level
SR_ATR_PERIOD = 14
SR_MIN_TOUCHES = 2           # prefer levels touched at least this often; weaker ones are only used if a side has none
# Rolling window for the Hurst exponent, in spine rows. The `hurst` package REFUSES series
# shorter than 100 points (raises ValueError) -- with the old window of 60 every call failed
# silently into the "return 0.5" fallback, so hurst_exponent was a constant 0.5 in both
# training and inference. Must equal HURST_WINDOW in lstm_hurst_prepare.py.
HURST_WINDOW = 100
# Regime thresholds for sr_regime. The simplified R/S estimator is biased upward on short
# windows: a pure random walk of 100 points gives H ~ 0.58 (5-95% range ~0.38-0.84), so
# 0.55 would label almost everything "trending". Tune to taste.
SR_HURST_TRENDING = 0.70         # H >= this  -> "trending"       (levels break more easily)
SR_HURST_MEAN_REVERTING = 0.45   # H <= this  -> "mean_reverting" (levels tend to hold)

TARGET_COLUMN = "target_volume_T_plus_1"
PREDICTION_COLUMN = "target_volume_T_plus_1_pred"

RANDOM_SEED = 42
