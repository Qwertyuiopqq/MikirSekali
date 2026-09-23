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
    xgboost_models_folder: str = "../../models/XGBoost"
    xgboost_model_filename_tpl: str = "finetuned_{company}_model.json"

    # ---- outputs ------------------------------------------------------
    output_folder: str = "../../data/output"
    mcs_raw_csv: str = "MCS_raw.csv"
    mcs_report_csv: str = "MCS_report.csv"
    mcs_predict_csv: str = "MCS_predict.csv"
    mcs_health_csv: str = "MCS_health.csv"

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

TARGET_COLUMN = "target_volume_T_plus_1"
PREDICTION_COLUMN = "target_volume_T_plus_1_pred"

RANDOM_SEED = 42
