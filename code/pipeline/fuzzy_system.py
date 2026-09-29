"""
fuzzy_system.py
----------------
The fuzzy business-health scoring engine. Pure logic, no I/O, so it's
easy to unit-test in isolation.

Three modes:
  - 'real'             : sentiment + trend + stability only
                          -> health_score_real
  - 'predict'           : same + XGBoost volume prediction contribution
                          -> health_score_predict
  - 'predict_breakout'  : same as 'predict' + LSTM-Hurst breakout
                          probability contribution (only meaningful on
                          rows where lstm_hurst_predictor.py actually
                          produced a breakout_probability; NaN there
                          falls back to treating that contribution as 0)
"""
import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:  # running as a plain script, not as a package
    import config


class BusinessHealthFuzzySystem:
    def __init__(self, weights_real=None, weights_predict=None):
        self.weights_real = weights_real or config.FUZZY_WEIGHTS_REAL
        self.weights_predict = weights_predict or config.FUZZY_WEIGHTS_PREDICT

    # ---- individual fuzzification functions ---------------------------
    @staticmethod
    def _fuzzify_sentiment(sentiment_score):
        return np.clip((sentiment_score + 0.5), 0, 1)

    @staticmethod
    def _fuzzify_trend(return_pct, ma_7, ma_30):
        fuzzy_return = np.clip((return_pct + 0.02) / 0.05, 0, 1)
        trend_multiplier = np.where(ma_7 > ma_30, 1.0, 0.5)
        return np.clip(fuzzy_return * trend_multiplier, 0, 1)

    @staticmethod
    def _fuzzify_stability(volatility, max_acceptable_volatility=0.05):
        return np.clip(1 - (volatility / max_acceptable_volatility), 0, 1)

    @staticmethod
    def _fuzzify_xgboost(xgb_prediction, max_expected_value=1.0):
        return np.clip(xgb_prediction / max_expected_value, 0, 1)

    @staticmethod
    def _fuzzify_breakout(breakout_probability):
        # Already a 0-1 probability out of the model's sigmoid; just clip
        # for safety and treat missing (row too short a history, model/
        # scaler unavailable, etc.) as "no signal" rather than dropping
        # the row.
        return np.clip(pd.Series(breakout_probability).fillna(0.0), 0, 1)

    # ---- public API -----------------------------------------------------
    def calculate_health_score(self, df: pd.DataFrame, mode: str = "real",
                                xgb_col: str = None, breakout_col: str = None) -> pd.Series:
        """
        Compute the 0-100 health score for every row of `df`.

        mode='real'             -> ignores xgb_col/breakout_col entirely
        mode='predict'          -> requires xgb_col
        mode='predict_breakout' -> requires xgb_col and breakout_col
        """
        if mode == "predict_breakout":
            weights = config.FUZZY_WEIGHTS_PREDICT_BREAKOUT
        elif mode == "predict":
            weights = self.weights_predict
        else:
            weights = self.weights_real

        f_sent = self._fuzzify_sentiment(df["daily_news_sentiment"])
        f_trend = self._fuzzify_trend(df["daily_return_pct"], df["MA_7_Close"], df["MA_30_Close"])
        f_stab = self._fuzzify_stability(df["volatility_7d"])

        total_score = (
            (f_sent * weights["sentiment"]) +
            (f_trend * weights["trend"]) +
            (f_stab * weights["stability"])
        )

        if mode in ("predict", "predict_breakout") and xgb_col is not None:
            f_xgb = self._fuzzify_xgboost(df[xgb_col])
            total_score = total_score + (f_xgb * weights["xgboost"])

        if mode == "predict_breakout" and breakout_col is not None:
            f_breakout = self._fuzzify_breakout(df[breakout_col]).values
            total_score = total_score + (f_breakout * weights["breakout"])

        return np.round(total_score * 100, 2)
