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

# Folder this file lives in (= pipeline/). Relative paths in `Paths.health_csv()` / `Paths.resolve()`
# are anchored here, so they mean the same thing no matter which folder you launch Python from.
_PIPELINE_DIR = Path(__file__).resolve().parent


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
    # Optional: breakout-probability models (written by lstm_hurst_train.py, read by lstm_hurst_predictor.py).
    # Layout:
    #
    #   <lstm_hurst_model_path>/
    #       _POOLED/  lstm_hurst_best_params.json     <- ONE model for every company (the default; the features
    #                 [best_lstm_hurst_model.pth]        are scale-free, so companies can share it). The .pth is
    #       ASSA/     ...                                only there when the LSTM beat the logistic model.
    #       (optional per-company sub-folders, name = symbol without ".JK", override the pooled model)
    #
    # A company that has neither its own folder nor a usable pooled model is skipped gracefully -- its
    # breakout_probability stays NaN -- instead of failing the whole pipeline (same pattern as the
    # optional sentiment file in enrichment.py). Models written by the OLD trainer (per-company, raw Close
    # feature, calendar-day windows) are recognised and skipped: they must be retrained.
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

    # ---- WHERE MCS_health.csv LIVES ---------------------------------------
    # >>> EDIT THIS LINE to the real location of your MCS_health.csv. <<<
    # It is the single place that decides the file for BOTH ends:
    #   * `python pipeline.py score` WRITES the final health table here, and
    #   * `website/server.py` READS (and serves) it from here.
    # Absolute path, or relative to this pipeline/ folder (e.g. "../../data/output/MCS_health.csv").
    # Leave it "" to fall back to <output_folder>/<mcs_health_csv> (the old behaviour).
    # CLI override: --mcs-health-csv-path <path>   (server.py also still accepts --csv <path>)
    mcs_health_csv_path: str = "../../data/output/MCS_health.csv"

    def output_path(self, filename: str) -> str:
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        return str(Path(self.output_folder) / filename)

    def resolve(self, p: str) -> str:
        """Absolute form of a path from this class (relative ones are anchored at pipeline/)."""
        q = Path(p).expanduser()
        if not q.is_absolute():
            q = _PIPELINE_DIR / q
        return str(q.resolve())

    def health_csv(self) -> str:
        """Absolute path of MCS_health.csv -- see `mcs_health_csv_path` above."""
        return self.resolve(self.mcs_health_csv_path or str(Path(self.output_folder) / self.mcs_health_csv))


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

# --- How the five fuzzy components are fed (see fuzzy_system.py) --------------------------------
# Missing-signal rule: a component with no signal on a row (NaN / not applicable) is LEFT OUT of that
# row's score and the remaining weights are renormalised, instead of being scored as 0 or 1.
#
# 1) SENTIMENT. enrichment.py invents RANDOM sentiment when news_sentiment.csv is missing; that noise
#    must not move a health score. False = rows flagged `sentiment_is_mock` leave the sentiment
#    component out. True = score on the noise anyway (old behaviour; only sensible for UI demos).
SCORE_USES_MOCK_SENTIMENT = False

# 2) XGBOOST volume outlook = "will volume rise or fall?". The raw forecast LEVEL is not comparable to actual
#    volume (on the 10-company dataset the model runs ~2x low, and by a different amount on each weekday), so a
#    forecast is judged against the model's OWN earlier forecasts for the same weekday:
#      ratio = forecast / median(the last XGB_NORM_WINDOW forecasts made on the same weekday)
#      ratio == 1 -> 0.5 (no change) | ratio >= FULL_SCALE -> 1.0 (rising) | ratio <= 1/FULL_SCALE -> 0.0 (falling)
#    Rows where the forecast is about a non-trading day (Fri/Sat -> weekend, Sun/holiday rows) have no usable
#    outlook and are left out of the score. 8 weeks / 4x: on the dataset the membership then averages 0.48 on every
#    weekday and is pinned at 0/1 on ~7% of rows (with a 4-week window and 2.5x it was 12-17%).
XGB_NORM_WINDOW = 8
XGB_NORM_MIN_PERIODS = 3
XGB_FULL_SCALE_RATIO = 4.0

# 3) BREAKOUT = LSTM-Hurst x support/resistance, SIGNED (0.5 = neutral):
#      price broke resistance -> + (LSTM breakout probability x Hurst trend-persistence)
#      price hit support      -> - ((1 - breakout probability) x Hurst trend-persistence)
#    "Hurst trend-persistence" = (H - SR_HURST_MEAN_REVERTING) / (SR_HURST_TRENDING - SR_HURST_MEAN_REVERTING)
#    clipped to 0..1: mean-reverting regime -> 0 (levels hold, no credit/penalty), trending -> 1 (levels give way).
#    The nearest S/R line is only ~3-4% from price, so a "hit" on ANY line fires on a quarter of all days (on the
#    10-company dataset: 3 touches -> a support hit is "active" on 29% of trading days, 4 touches -> 16%, 5 -> 9%).
#    Events therefore only count for levels touched at least SR_SIGNAL_MIN_TOUCHES times (or the 1-year extreme).
SR_SIGNAL_MIN_TOUCHES = 4
SR_TOUCH_TOL = 0.0         # "hit support": today's Low <= yesterday's support line * (1 + this)
SR_BREAK_MARGIN = 0.0      # "broke resistance": today's Close > yesterday's resistance line * (1 + this)
SR_EVENT_PERSIST_DAYS = 3  # an event fades linearly over this many trading days (1.0, 0.67, 0.33, then gone)

# ---- XGBoost forecast target ---------------------------------------------------------------------------------
# What target_volume_T_plus_1 means (spine_builder.py builds it, train_xgboost_transfer.py learns it):
#   "next_trading_day"  volume of the next SESSION in which the stock traded (Friday -> Monday, holiday -> the day after).
#   "next_calendar_day" the OLD definition: volume of tomorrow's calendar day, which is 0 on weekends and holidays.
# The old one made 37% of the training targets zero (28.5% weekends + 8.2% weekday holidays/no-trade days), which
# pulled every forecast down ~2x (median actual/forecast = 1.94 on the 10-company dataset) and made Fri/Sat forecasts
# ~0. A model trained on "next_trading_day" has neither problem. A model WITHOUT training metadata
# (xgb_training_meta.json, written by the new trainer) is assumed to be XGB_LEGACY_TARGET_MODE.
XGB_TARGET_MODE = "next_trading_day"
XGB_LEGACY_TARGET_MODE = "next_calendar_day"
TARGET_MAX_GAP_DAYS = 7          # next session further away than this (a suspension) -> no target for that row
XGB_META_FILENAME = "xgb_training_meta.json"

# ---- Breakout-probability model (lstm_features.py / lstm_hurst_train.py / lstm_hurst_predictor.py) ----------
# P(Close exceeds its LSTM_YEARLY_WINDOW-trading-day high within the next LSTM_HORIZON trading days).
# Everything is measured in TRADING days (the old version used calendar rows: its "252-day" window was ~8
# months, its "30 days" ~21 sessions, and the Hurst window was 37% weekend/holiday forward-fills).
LSTM_FEATURE_VERSION = 2          # models without this version (the old trainer) are skipped by the predictor
LSTM_SEQ_LENGTH = 30              # sessions of history the LSTM sees
LSTM_HORIZON = 21                 # label horizon, trading days (~ one month)
LSTM_YEARLY_WINDOW = 252          # resistance / support = rolling max / min Close over this many sessions
LSTM_YEARLY_MIN_PERIODS = 60
LSTM_VOL_WINDOW = 20              # sessions for the volatility that scales the distances
LSTM_MAX_ABS_LOG_RET = 0.5        # a one-session |log return| above this is an unadjusted split: the price series is chain-linked
# Scale-free inputs only (the old model fed the raw Close level, which was up to 24 standard deviations out of its
# training range for ELPI). dist_*_sigma = ln(level / Close) in units of the horizon's volatility, the quantity a
# first-passage probability depends on; hurst_c = Hurst exponent clipped to 0..1.
LSTM_FEATURES = ["dist_res_sigma", "dist_sup_sigma", "range_pos", "log_ret", "hurst_c"]
# Predictor gate: skip a trained model whose own hold-out report says it did NOT beat the trivial baselines
# (lstm_hurst_best_params.json -> beats_baseline == false). False = use it anyway.
LSTM_USE_ONLY_IF_BEATS_BASELINE = True
LSTM_POOLED_DIRNAME = "_POOLED"

# Artifact filenames (inside <lstm_hurst_model_path>/<_POOLED or COMPANY>/).
LSTM_MODEL_FILENAME = "best_lstm_hurst_model.pth"
LSTM_PARAMS_FILENAME = "lstm_hurst_best_params.json"   # holds the scaler, logistic coefficients, calibration, metrics
LSTM_SCALER_FILENAME = "lstm_hurst_scaler.joblib"      # old trainer only (the scaler now lives in the params file)

# Support / resistance lines (pipeline/support_resistance.py).
SR_PIVOT_WINDOW = 5          # bars on each side needed to confirm a swing high/low
SR_LOOKBACK_TRADING_DAYS = 252   # ~1 trading year of pivots are considered
SR_CLUSTER_ATR_MULT = 0.6    # pivots closer than this many ATRs are merged into one level
SR_ATR_PERIOD = 14
SR_MIN_TOUCHES = 2           # prefer levels touched at least this often; weaker ones are only used if a side has none
# Rolling window for the Hurst exponent, in TRADING DAYS (it is computed on the session closes only; the old
# version ran on the calendar spine, where 37% of the window was flat weekend/holiday fill, and 1.7% of the
# estimates came out above 1). The `hurst` package REFUSES series shorter than 100 points (raises
# ValueError) -- with the old window of 60 every call failed silently into the "return 0.5" fallback, so
# hurst_exponent was a constant 0.5 in both training and inference.
HURST_WINDOW = 100
# Regime thresholds for sr_regime. The simplified R/S estimator is biased upward on short
# windows: a pure random walk of 100 points gives H ~ 0.58 (5-95% range ~0.38-0.84), so
# 0.55 would label almost everything "trending". Tune to taste.
SR_HURST_TRENDING = 0.70         # H >= this  -> "trending"       (levels break more easily)
SR_HURST_MEAN_REVERTING = 0.45   # H <= this  -> "mean_reverting" (levels tend to hold)

TARGET_COLUMN = "target_volume_T_plus_1"
PREDICTION_COLUMN = "target_volume_T_plus_1_pred"

RANDOM_SEED = 42
