"""
input_validator.py
-------------------
Checks a user-supplied CSV against the schema of MCS_features.csv (the
"master calendar spine", enriched -- the same shape the real pipeline
feeds into XGBoost and the fuzzy system) before anything else runs, so
a malformed file fails fast with a clear message instead of a
confusing pandas KeyError three stages in.
"""
import os
import sys
from dataclasses import dataclass, field
from typing import List

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline import config

BASE_COLUMNS = ["Date", "symbol"]
# Needed by the fuzzy health system on top of the XGBoost feature list.
FUZZY_EXTRA_COLUMNS = ["daily_news_sentiment", "daily_return_pct", "MA_7_Close", "MA_30_Close", "volatility_7d"]


@dataclass
class ValidationResult:
    ok: bool
    missing_columns: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    has_target: bool = False

    def raise_if_invalid(self):
        if not self.ok:
            raise ValueError(
                "Input CSV does not match the MCS_features master calendar "
                f"spine format. Missing required columns: {self.missing_columns}"
            )


def validate_spine_format(df: pd.DataFrame, feature_cols=None,
                           target_col: str = config.TARGET_COLUMN) -> ValidationResult:
    """
    Confirms `df` has every column the XGBoost models and the fuzzy
    health system need. Does NOT require target_col (the ground-truth
    next-day volume) -- its absence is a warning, not a failure, since
    pure future/forecast rows won't have it yet.
    """
    feature_cols = feature_cols or config.XGBOOST_FEATURES
    required = list(dict.fromkeys(BASE_COLUMNS + feature_cols + FUZZY_EXTRA_COLUMNS))

    missing = [c for c in required if c not in df.columns]
    warnings = []

    if "Date" not in missing:
        try:
            pd.to_datetime(df["Date"])
        except Exception as e:
            missing.append("Date (unparseable)")
            warnings.append(f"'Date' column could not be parsed as dates: {e}")

    has_target = target_col in df.columns
    if not has_target:
        warnings.append(
            f"'{target_col}' not found in input -> treating this as future/"
            "forecast-only data (no ground truth). MAE/MAPE will be skipped; "
            "you'll still get predictions + health_score_predict."
        )
    elif df[target_col].isna().all():
        has_target = False
        warnings.append(
            f"'{target_col}' is present but entirely empty -> MAE/MAPE will be skipped."
        )

    return ValidationResult(ok=(len(missing) == 0), missing_columns=missing,
                             warnings=warnings, has_target=has_target)
