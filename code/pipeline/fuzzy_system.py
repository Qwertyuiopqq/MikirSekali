"""
fuzzy_system.py
----------------
The fuzzy business-health scoring engine. Pure logic, no I/O, so it's
easy to unit-test in isolation (see testing/test_fuzzy_system.py).

Every component is turned into a 0..1 membership (0.5 = "no change /
neutral" wherever a direction exists) and the health score is the
weighted average of the memberships x 100:

  sentiment  FinBERT  P(positive) - P(negative)  of the company's news, -1..+1
             (+-0.5 already saturates the membership). Only days WITH news count: random
             mock sentiment and days with daily_news_count == 0 have no signal.
  trend      today's return, halved while the 7-day MA is below the 30-day MA.
  stability  7-day volatility: 0% -> 1.0, >= 5% -> 0.0.
  xgboost    VOLUME OUTLOOK -- will volume rise or fall? Tomorrow's forecast is
             judged against the model's OWN earlier forecasts for the same weekday
             (median of the last `config.XGB_NORM_WINDOW` Mondays / Tuesdays / ...):
                 ratio = forecast / norm
                 ratio == 1 -> 0.5 | ratio >= XGB_FULL_SCALE_RATIO -> 1.0 (rising)
                 ratio <= 1 / XGB_FULL_SCALE_RATIO -> 0.0 (falling)
             Why not forecast vs actual volume: the raw forecast level runs ~2x low
             (log-space training), so "forecast / actual volume" would read as
             "falling" on almost every day. The old code divided the forecast by 1.0
             and clipped to 1 -- volumes are in the millions, so it was ALWAYS 1.0.
  breakout   LSTM-Hurst x support/resistance, SIGNED, neutral = 0.5:
                 price broke resistance -> UP   : + P(breakout) x Hurst-persistence
                 price hit support      -> DOWN : - (1 - P(breakout)) x Hurst-persistence
             P(breakout) is the LSTM's chance that Close exceeds its 252-day
             high (`yearly_resistance`, 252 calendar rows ~ 8 months) within 30 days; Hurst-persistence is 0 for a mean-reverting
             regime (levels hold) and 1 for a trending one (levels give way), see
             `hurst_persistence`. Events come from support_resistance.add_sr_events.
             (The LSTM only models the UPSIDE break, so on the support side it can
             only damp the penalty via 1 - P. A true P(break support) would need a
             second LSTM label.)

Three modes (which components take part):
  'real'             sentiment + trend + stability                   -> health_score_real
  'predict'          + xgboost                                       -> health_score_predict
  'predict_breakout' + xgboost + breakout (when the LSTM-Hurst model
                     produced a probability for that company)        -> health_score_predict

Missing-signal rule: a component that has no signal on a row (NaN, an
XGBoost forecast about a non-trading day, sentiment that is mock noise, no
LSTM model for the company, ...) is LEFT OUT of that row's score and the
remaining weights are renormalised. trend and stability are the core price
inputs: without them a row has no score (NaN).
"""
import numpy as np
import pandas as pd

try:
    from . import config
    from .support_resistance import add_sr_events, SR_EVENT_COLUMNS
except ImportError:  # running as a plain script, not as a package
    import config
    from support_resistance import add_sr_events, SR_EVENT_COLUMNS

CORE_COMPONENTS = ("trend", "stability")
FORECAST_COLUMNS = ["xgb_applicable", "xgb_norm"]


# ---- volume-forecast helpers (used by the score AND exposed in MCS_health.csv) -------------
def xgb_applicable(df: pd.DataFrame) -> pd.Series:
    """
    True where the XGBoost forecast describes a REAL next trading day: today traded
    (Volume > 0) and the next calendar day is Mon-Fri.

    The model is trained on the next CALENDAR day's volume, so on Fri/Sat rows it
    (correctly) forecasts ~0 for the weekend, and on Sun/holiday rows it works from
    zero-volume features. Scoring those forecasts made health_score_predict drop by
    ~26 points on every Friday and Saturday.
    """
    dow = (df["day_of_week"] if "day_of_week" in df.columns
           else pd.to_datetime(df["Date"]).dt.dayofweek)
    traded = (df["Volume"] > 0) if "Volume" in df.columns else pd.Series(True, index=df.index)
    ok = traded.to_numpy(dtype=bool) & dow.isin([0, 1, 2, 3]).to_numpy(dtype=bool)
    return pd.Series(ok, index=df.index)


def forecast_norm(df: pd.DataFrame, pred_col: str, applicable=None,
                  window: int = None, min_periods: int = None) -> pd.Series:
    """
    The model's "usual" forecast to compare today's forecast with: per symbol AND per weekday of the
    forecasting row, the trailing MEDIAN of the model's own forecasts over its last `window` usable
    rows (= the last `window` Mondays, Tuesdays, ...). NaN on rows that are not applicable.

    Causal (row t only sees forecasts up to t). Median because volume forecasts are spiky. Because the
    forecast is compared with ITS OWN history for the same weekday, the model's level bias cancels --
    including its weekday-dependent part (on this dataset the Monday-row forecasts run ~0.5 log-units
    above Tue-Thu ones, so a pooled norm made every Monday look like "volume rising").
    """
    window = config.XGB_NORM_WINDOW if window is None else window
    min_periods = config.XGB_NORM_MIN_PERIODS if min_periods is None else min_periods
    ok = (xgb_applicable(df) if applicable is None else pd.Series(applicable, index=df.index)).to_numpy(dtype=bool)

    n = len(df)
    codes = pd.factorize(df["symbol"], sort=True)[0] if "symbol" in df.columns else np.zeros(n, dtype=int)
    dates = pd.to_datetime(df["Date"]) if "Date" in df.columns else None
    dow = (df["day_of_week"].to_numpy(dtype=float) if "day_of_week" in df.columns
           else dates.dt.dayofweek.to_numpy(dtype=float))
    dow = np.nan_to_num(dow, nan=0.0).astype(np.int64)
    when = dates.to_numpy("datetime64[ns]").astype("int64") if dates is not None else np.arange(n)
    vals = np.where(ok, df[pred_col].to_numpy(dtype=float), np.nan)

    key = codes.astype(np.int64) * 7 + dow                   # one series per (symbol, weekday)
    out = np.full(n, np.nan)
    order = np.lexsort((when, key))
    cuts = np.flatnonzero(np.diff(key[order])) + 1
    for gpos in np.split(order, cuts):                       # oldest first
        s = pd.Series(vals[gpos], index=gpos)
        norm = s.dropna().rolling(window, min_periods=min_periods).median()
        out[gpos] = norm.reindex(gpos).to_numpy()            # only rows that had a usable forecast get a norm
    return pd.Series(out, index=df.index)


def add_forecast_norm(df: pd.DataFrame, pred_col: str = None) -> pd.DataFrame:
    """Adds `xgb_applicable` and `xgb_norm` if they are not there yet (returns a copy)."""
    pred_col = pred_col or config.PREDICTION_COLUMN
    out = df.copy()
    if "xgb_applicable" not in out.columns:
        out["xgb_applicable"] = xgb_applicable(out)
    if "xgb_norm" not in out.columns:
        out["xgb_norm"] = forecast_norm(out, pred_col, out["xgb_applicable"])
    return out


class BusinessHealthFuzzySystem:
    def __init__(self, weights_real=None, weights_predict=None, weights_predict_breakout=None):
        self.weights_real = weights_real or config.FUZZY_WEIGHTS_REAL
        self.weights_predict = weights_predict or config.FUZZY_WEIGHTS_PREDICT
        self.weights_predict_breakout = weights_predict_breakout or config.FUZZY_WEIGHTS_PREDICT_BREAKOUT

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
    def _fuzzify_xgboost(forecast, norm, applicable=None):
        """Volume outlook, 0.5 = forecast equals the model's recent norm (see module docstring)."""
        forecast, norm = pd.Series(forecast), pd.Series(norm, index=pd.Series(forecast).index)
        log_ratio = np.log(np.maximum(forecast, 1.0) / np.maximum(norm, 1.0))
        f = np.clip(0.5 + 0.5 * log_ratio / np.log(config.XGB_FULL_SCALE_RATIO), 0.0, 1.0)
        if applicable is not None:
            f = f.where(np.asarray(applicable, dtype=bool))
        return f

    @staticmethod
    def hurst_persistence(hurst):
        """
        0..1 "how strongly does this regime push through levels": 0 at/below the mean-reverting
        threshold (levels hold), 1 at/above the trending threshold (levels give way), linear between.
        Same thresholds as the regime labels, so the score agrees with `sr_regime` on the website.
        """
        lo, hi = config.SR_HURST_MEAN_REVERTING, config.SR_HURST_TRENDING
        return np.clip((hurst - lo) / (hi - lo), 0.0, 1.0)

    @classmethod
    def _fuzzify_breakout(cls, prob, hurst, up, down):
        """
        Signed LSTM-Hurst x S/R membership in 0..1 (0.5 = no event).
        `up` / `down` are the 0..1 strengths of 'broke resistance' / 'hit support'.
        """
        p = np.clip(prob, 0.0, 1.0)
        h = cls.hurst_persistence(hurst)
        signed = np.clip(p * h * up - (1.0 - p) * h * down, -1.0, 1.0)
        return 0.5 + 0.5 * signed

    # ---- components ---------------------------------------------------
    def _weights(self, mode: str) -> dict:
        return {"predict_breakout": self.weights_predict_breakout,
                "predict": self.weights_predict}.get(mode, self.weights_real)

    def memberships(self, df: pd.DataFrame, mode: str = "real", xgb_col: str = None,
                    breakout_col: str = None, hurst_col: str = "hurst_exponent") -> pd.DataFrame:
        """One 0..1 membership column per active component; NaN = this row has no signal for it."""
        out = pd.DataFrame(index=df.index)

        sentiment = self._fuzzify_sentiment(df["daily_news_sentiment"])
        if not config.SCORE_USES_MOCK_SENTIMENT and "sentiment_is_mock" in df.columns:
            mock = df["sentiment_is_mock"].astype("boolean").fillna(False).to_numpy(dtype=bool)
            sentiment = sentiment.where(~mock)          # random mock noise is not evidence
        if "daily_news_count" in df.columns:
            no_news = (df["daily_news_count"] <= 0).to_numpy(dtype=bool)   # enrichment fills days without news with 0 / 0
            sentiment = sentiment.where(~no_news)       # no news is no evidence either (not a "neutral" reading)
        out["sentiment"] = sentiment
        out["trend"] = self._fuzzify_trend(df["daily_return_pct"], df["MA_7_Close"], df["MA_30_Close"])
        out["stability"] = self._fuzzify_stability(df["volatility_7d"])

        if mode in ("predict", "predict_breakout") and xgb_col is not None:
            ok = df["xgb_applicable"] if "xgb_applicable" in df.columns else xgb_applicable(df)
            norm = df["xgb_norm"] if "xgb_norm" in df.columns else forecast_norm(df, xgb_col, ok)
            out["xgboost"] = self._fuzzify_xgboost(df[xgb_col], norm, ok)

        if mode == "predict_breakout" and breakout_col is not None:
            ev = df if set(SR_EVENT_COLUMNS) <= set(df.columns) else add_sr_events(df)
            hurst = df[hurst_col] if hurst_col in df.columns else pd.Series(np.nan, index=df.index)
            out["breakout"] = self._fuzzify_breakout(df[breakout_col], hurst,
                                                     ev["sr_break_up"], ev["sr_hit_support"])
        return out

    @staticmethod
    def _effective_weights(m: pd.DataFrame, weights: dict) -> pd.DataFrame:
        """Per-row weights of the components that are PRESENT, renormalised to sum to 1."""
        w = pd.Series({c: float(weights[c]) for c in m.columns})
        present = m.notna().mul(w, axis=1)
        total = present.sum(axis=1).where(lambda t: t > 0)
        return present.div(total, axis=0)

    def explain(self, df: pd.DataFrame, mode: str = "predict_breakout", xgb_col: str = None,
                breakout_col: str = None) -> dict:
        """
        Per-row breakdown that always adds up to the headline score:
          membership  0..1 per component (NaN = left out)
          weight      effective (renormalised) weight per component
          points      membership x weight x 100  -> sums to `score`
          score       the 0..100 health score
        """
        m = self.memberships(df, mode, xgb_col, breakout_col)
        eff = self._effective_weights(m, self._weights(mode))
        points = m.fillna(0.0).mul(eff) * 100.0
        core = m[[c for c in CORE_COMPONENTS if c in m.columns]].notna().all(axis=1)
        score = points.sum(axis=1, min_count=1).where(core).round(2)
        return {"membership": m, "weight": eff, "points": points, "score": score}

    # ---- public API -----------------------------------------------------
    def calculate_health_score(self, df: pd.DataFrame, mode: str = "real",
                                xgb_col: str = None, breakout_col: str = None) -> pd.Series:
        """
        Compute the 0-100 health score for every row of `df`.

        mode='real'             -> ignores xgb_col/breakout_col entirely
        mode='predict'          -> uses xgb_col (if given)
        mode='predict_breakout' -> uses xgb_col and breakout_col (if given)
        """
        return self.explain(df, mode, xgb_col, breakout_col)["score"]

    def score_frame(self, df: pd.DataFrame, xgb_col: str = None,
                    breakout_col: str = "breakout_probability") -> pd.DataFrame:
        """
        Everything the website and MCS_health.csv need, from one call: adds `xgb_applicable`,
        `xgb_norm`, the S/R event columns (when missing), `health_score_real` and
        `health_score_predict`. Used by pipeline.py (Stage 6) AND website/server.py, so both
        always apply the same formulas.
        """
        xgb_col = xgb_col or config.PREDICTION_COLUMN
        out = df.copy()
        has_xgb = xgb_col in out.columns
        has_bp = breakout_col in out.columns
        if has_xgb:
            out = add_forecast_norm(out, xgb_col)
        if has_bp and not set(SR_EVENT_COLUMNS) <= set(out.columns):
            out = add_sr_events(out)
        out["health_score_real"] = self.calculate_health_score(out, "real")
        if has_xgb:
            out["health_score_predict"] = self.calculate_health_score(
                out, "predict_breakout" if has_bp else "predict", xgb_col, breakout_col if has_bp else None)
        return out
