"""
lstm_features.py
-----------------
Everything the breakout-probability model needs that is NOT torch: feature engineering, labels, sequences,
chronological splits with an embargo, logistic baseline, calibration and metrics. numpy / pandas / scikit-learn only.

lstm_hurst_train.py (training) and lstm_hurst_predictor.py (inference) both import THIS module, so the features
the model is trained on and the features it is scored on cannot drift apart (the classic silent bug).

What is predicted
    P(Close exceeds its LSTM_YEARLY_WINDOW-session high within the next LSTM_HORIZON sessions)
    yearly_resistance[t] = max(Close[t-251 .. t])  (trading sessions, min LSTM_YEARLY_MIN_PERIODS)
    label[t]            = 1 if max(Close[t+1 .. t+LSTM_HORIZON]) > yearly_resistance[t]
    Everything is on TRADING-DAY rows (Volume > 0). The old version ran on the calendar spine, where the
    "252-day" window was ~8 months, "30 days" was ~21 sessions, and 37% of the Hurst window was flat
    weekend/holiday fill.

Features (config.LSTM_FEATURES; all scale-free, so one model can serve every company)
    dist_res_sigma  ln(resistance / Close) in units of the horizon's volatility  = how many "sigmas" away the high is
    dist_sup_sigma  ln(Close / support)    in the same units
    range_pos       (Close - support) / (resistance - support), 0..1
    log_ret         one-session log return, clipped to +-0.4
    hurst_c         Hurst exponent of the last HURST_WINDOW sessions, clipped to 0..1
  The old model was fed the raw Close level (z-scored with its training period): for ELPI it was up to 24 standard
  deviations out of range in the test period. Price discontinuities (|log return| > LSTM_MAX_ABS_LOG_RET, i.e.
  unadjusted splits such as TMAS in 2023-05) are chain-linked: earlier prices are rescaled so the jump disappears.
"""
import json
import math
import os

import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:  # running as a plain script, not as a package
    import config

FRACTAL_EXPORT = ["yearly_resistance", "yearly_support", "dist_to_resistance", "dist_to_support", "hurst_exponent"]
_HURST_ESTIMATOR = None


# ---- Hurst ----------------------------------------------------------------------------------------------------
def set_hurst_estimator(fn):
    """Replace the estimator (tests, or another implementation). fn(window_of_closes) -> float."""
    global _HURST_ESTIMATOR
    _HURST_ESTIMATOR = fn


def get_hurst_estimator():
    """The `hurst` package's simplified R/S estimator on price series (the one the old pipeline used), or None."""
    global _HURST_ESTIMATOR
    if _HURST_ESTIMATOR is None:
        try:
            from hurst import compute_Hc
        except ImportError:
            return None

        def _estimate(window):
            try:
                H, _, _ = compute_Hc(window, kind="price", simplified=True)
                return float(H)
            except Exception:
                return np.nan                       # NOT 0.5: a failed estimate must not look like a measurement
        _HURST_ESTIMATOR = _estimate
    return _HURST_ESTIMATOR


def rolling_hurst(close, window=None, estimator=None):
    """Rolling Hurst exponent over `window` consecutive closes; NaN until the window is full or when it fails."""
    window = config.HURST_WINDOW if window is None else window
    est = estimator or get_hurst_estimator()
    close = np.asarray(close, float)
    out = np.full(len(close), np.nan)
    if est is None:
        return out
    for i in range(window - 1, len(close)):
        out[i] = est(close[i - window + 1:i + 1])
    return out


# ---- prices ---------------------------------------------------------------------------------------------------
def chain_link(close, max_abs_log_ret=None):
    """
    Remove price discontinuities (unadjusted splits): wherever |ln(c[j]/c[j-1])| exceeds the threshold, every
    EARLIER price is rescaled so the series is continuous there. Returns (adjusted close, number of jumps).
    """
    thr = config.LSTM_MAX_ABS_LOG_RET if max_abs_log_ret is None else max_abs_log_ret
    c = np.asarray(close, float).copy()
    n_jumps = 0
    for j in range(1, len(c)):
        if c[j - 1] > 0 and c[j] > 0 and abs(math.log(c[j] / c[j - 1])) > thr:
            c[:j] *= c[j] / c[j - 1]
            n_jumps += 1
    return c, n_jumps


def symbol_frame(g, with_label=False, estimator=None):
    """
    Trading-day feature frame for ONE company. `g` = its calendar-spine rows (Date, Close, Volume). Columns:
    Date, close_adj, yearly_resistance, yearly_support, dist_to_resistance, dist_to_support, hurst_exponent (raw),
    the model features (config.LSTM_FEATURES) and, with `with_label`, is_breakout (NaN where the future is unknown).
    `.attrs["n_jumps"]` = number of chain-linked price discontinuities.
    """
    t = g.loc[g["Volume"] > 0, ["Date", "Close"]].dropna().sort_values("Date")
    t = t[t["Close"] > 0].reset_index(drop=True)
    close, n_jumps = chain_link(t["Close"].to_numpy(float))
    s = pd.Series(close)
    H, YW, YMIN, VW = config.LSTM_HORIZON, config.LSTM_YEARLY_WINDOW, config.LSTM_YEARLY_MIN_PERIODS, config.LSTM_VOL_WINDOW

    log_ret = np.log(s / s.shift(1))
    res = s.rolling(YW, min_periods=YMIN).max()
    sup = s.rolling(YW, min_periods=YMIN).min()
    vol = log_ret.rolling(VW, min_periods=max(VW // 2, 2)).std()
    sigma = np.maximum(vol, 1e-4) * math.sqrt(H)                      # horizon volatility in log units
    dist_res, dist_sup = np.log(res / s), np.log(s / sup)
    span = res - sup
    hurst = rolling_hurst(close, estimator=estimator)

    out = pd.DataFrame({
        "Date": t["Date"].to_numpy(), "close_adj": close,
        "yearly_resistance": res.to_numpy(), "yearly_support": sup.to_numpy(),
        "dist_to_resistance": (res / s - 1.0).to_numpy(), "dist_to_support": (s / sup - 1.0).to_numpy(),
        "hurst_exponent": hurst,
        "dist_res_sigma": np.clip(dist_res / sigma, 0.0, 10.0).to_numpy(),
        "dist_sup_sigma": np.clip(dist_sup / sigma, 0.0, 10.0).to_numpy(),
        "range_pos": np.where(span > 0, (s - sup) / span, 0.5).astype(float),
        "log_ret": np.clip(log_ret, -0.4, 0.4).to_numpy(),
        "hurst_c": np.clip(hurst, 0.0, 1.0),
    })
    out.loc[res.isna().to_numpy(), "range_pos"] = np.nan
    if with_label:
        n = len(close)
        fut_max = np.full(n, np.nan)
        if n > H:
            windows = np.lib.stride_tricks.sliding_window_view(close, H)          # windows[i] = close[i .. i+H-1]
            fut_max[:n - H] = windows[1:n - H + 1].max(axis=1)                    # max(close[t+1 .. t+H])
        ok = ~np.isnan(fut_max) & res.notna().to_numpy()
        out["is_breakout"] = np.where(ok, (fut_max > res.to_numpy()).astype(float), np.nan)
    out.attrs["n_jumps"] = n_jumps
    return out


def build_symbol_frames(df, with_label=False, estimator=None, verbose=False):
    """{symbol: trading-day feature frame} for every company in the calendar-spine frame `df`."""
    frames = {}
    for sym, g in df.groupby("symbol"):
        frames[sym] = symbol_frame(g, with_label=with_label, estimator=estimator)
        if verbose and frames[sym].attrs.get("n_jumps"):
            print(f"   {sym}: {frames[sym].attrs['n_jumps']} price discontinuity(ies) chain-linked (unadjusted split?)")
    return frames


def calendar_ffill(cal_dates, table):
    """
    Look `table` (a frame with a 'Date' column and value columns) up on the calendar rows `cal_dates` (any order)
    and forward-fill in date order, so weekends/holidays inherit the last session's values -- the same convention
    the support/resistance lines use. Returns a frame with the SAME index as `cal_dates`.
    """
    cal = pd.DataFrame({"Date": pd.to_datetime(cal_dates).to_numpy(), "_pos": np.arange(len(cal_dates))}, index=cal_dates.index)
    merged = cal.sort_values("Date").merge(table.assign(Date=pd.to_datetime(table["Date"])), on="Date", how="left")
    value_cols = [c for c in table.columns if c != "Date"]
    merged[value_cols] = merged[value_cols].ffill()
    merged = merged.sort_values("_pos")
    return pd.DataFrame(merged[value_cols].to_numpy(), index=cal_dates.index, columns=value_cols)


def export_fractal_columns(df, frames):
    """Adds yearly_resistance / yearly_support / dist_to_* / hurst_exponent to the calendar frame (NaN where unknown)."""
    out = df.copy()
    for c in FRACTAL_EXPORT:
        out[c] = np.nan
    for sym, fr in frames.items():
        idx = out.index[out["symbol"] == sym]
        if len(idx):
            out.loc[idx, FRACTAL_EXPORT] = calendar_ffill(out.loc[idx, "Date"], fr[["Date"] + FRACTAL_EXPORT]).to_numpy()
    return out


# ---- samples, splits ------------------------------------------------------------------------------------------
def build_samples(frame, features=None, seq_len=None, with_label=True):
    """
    Sequences of `seq_len` consecutive sessions ending at each row. Returns dict(X (n, seq_len, f), y, end (row index
    of the last session), date, label_end). `label_end` = the date of the last price the label looks at (`horizon`
    sessions after the sequence ends; None without labels), which the split uses to PURGE samples whose label would
    reach into the next block. Windows containing a NaN feature (and, with labels, rows without a label) are dropped.
    """
    features = features or config.LSTM_FEATURES
    seq_len = seq_len or config.LSTM_SEQ_LENGTH
    n = len(frame)
    empty = {"X": np.empty((0, seq_len, len(features))), "y": np.empty(0), "end": np.empty(0, int),
             "date": np.empty(0, "datetime64[ns]"), "label_end": np.empty(0, "datetime64[ns]")}
    if n < seq_len:
        return empty
    X = frame[features].to_numpy(float)
    windows = np.transpose(np.lib.stride_tricks.sliding_window_view(X, seq_len, axis=0), (0, 2, 1))   # (n-seq+1, seq, f)
    end = np.arange(seq_len - 1, n)
    valid = ~np.isnan(windows).any(axis=(1, 2))
    y = frame["is_breakout"].to_numpy(float)[end] if with_label else None
    if with_label:
        valid &= ~np.isnan(y)
    dates = frame["Date"].to_numpy("datetime64[ns]")
    label_end = dates[np.minimum(end[valid] + config.LSTM_HORIZON, n - 1)] if with_label else None
    return {"X": windows[valid], "y": y[valid] if with_label else None, "end": end[valid],
            "date": dates[end[valid]], "label_end": label_end}


def split_cutoffs(dates, train_frac=0.6, val_frac=0.2, embargo_days=7):
    """
    ONE set of date cut-offs for every company (no company's future leaks into another's training):
        train: end date <= d1 | validation: d1 + embargo < end date <= d2 | test: end date > d2 + embargo
    plus the PURGE in `block_masks`: a sample whose label looks past its block's end is dropped. (A fixed calendar embargo
    cannot do that job: 21 consecutive SESSIONS span 29-36 calendar days normally, 48 for a thinly traded stock.)
    The small embargo only keeps near-duplicate neighbours (overlapping windows, adjacent labels) from straddling a boundary.
    """
    u = np.sort(pd.unique(pd.to_datetime(dates)))
    d1 = pd.Timestamp(u[max(int(len(u) * train_frac) - 1, 0)])
    d2 = pd.Timestamp(u[max(int(len(u) * (train_frac + val_frac)) - 1, 0)])
    return {"d1": d1, "d2": d2, "embargo": pd.Timedelta(days=int(embargo_days))}


def block_masks(dates, cuts, label_end=None):
    """
    Boolean masks {train, val, test} over samples with end dates `dates`. With `label_end` (the date of the last price each
    label uses) the blocks are PURGED: train labels end by d1, validation labels by d2, so no label ever uses a price
    from a later block.
    """
    d = pd.to_datetime(dates)
    train, val, test = np.asarray(d <= cuts["d1"]), np.asarray((d > cuts["d1"] + cuts["embargo"]) & (d <= cuts["d2"])), \
        np.asarray(d > cuts["d2"] + cuts["embargo"])
    if label_end is not None:
        le = pd.to_datetime(label_end)
        train &= np.asarray(le <= cuts["d1"])
        val &= np.asarray(le <= cuts["d2"])
    return {"train": train, "val": val, "test": test}


# ---- scaler, logistic model, calibration, metrics ----------------------------------------------------------------
def fit_scaler(rows):
    """Mean / std of the feature rows (the TRAIN block only). std 0 -> 1 so a constant feature cannot divide by zero."""
    rows = np.asarray(rows, float)
    mean, scale = rows.mean(axis=0), rows.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return {"mean": mean.tolist(), "scale": scale.tolist()}


def standardize(x, scaler):
    return (np.asarray(x, float) - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logit(Z, y, C=1.0):
    """Logistic regression on STANDARDIZED features Z. Returns {coef, intercept}."""
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=C, max_iter=1000).fit(np.asarray(Z, float), np.asarray(y, int))
    return {"coef": m.coef_.ravel().tolist(), "intercept": float(m.intercept_[0])}


def logit_scores(model, Z):
    """Raw log-odds of a fitted logistic model."""
    return np.asarray(Z, float) @ np.asarray(model["coef"]) + model["intercept"]


def fit_platt(z, y, C=1.0):
    """Platt scaling p = sigmoid(a*z + b) fitted on log-odds z (validation block). Identity if there is only one class."""
    z, y = np.asarray(z, float), np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return {"a": 1.0, "b": 0.0, "fitted": False}
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=C, max_iter=1000).fit(z.reshape(-1, 1), y)
    return {"a": float(m.coef_[0, 0]), "b": float(m.intercept_[0]), "fitted": True}


def apply_platt(z, platt):
    return sigmoid(platt["a"] * np.asarray(z, float) + platt["b"])


def binary_metrics(y, p, n_bins=10):
    """AUC (NaN if one class), Brier, log-loss, expected calibration error, and the plain counts."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    y, p = np.asarray(y, int), np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "base_rate": float(y.mean()) if len(y) else np.nan,
           "mean_p": float(p.mean()) if len(y) else np.nan}
    if not len(y):
        return {**out, "auc": np.nan, "brier": np.nan, "logloss": np.nan, "ece": np.nan}
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan
    out["brier"] = float(brier_score_loss(y, p))
    out["logloss"] = float(log_loss(y, p, labels=[0, 1]))
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    out["ece"] = float(sum(abs(y[bins == b].mean() - p[bins == b].mean()) * (bins == b).sum() for b in range(n_bins) if (bins == b).any()) / len(y))
    return out


# ---- params file ----------------------------------------------------------------------------------------------
def read_params(model_dir):
    path = os.path.join(model_dir, config.LSTM_PARAMS_FILENAME)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_params(model_dir, params):
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, config.LSTM_PARAMS_FILENAME), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, default=float)


def logit_probabilities(params, frame):
    """
    Probabilities of a saved LOGISTIC model for every row of a symbol frame whose features are complete
    (NaN elsewhere). No torch needed. Returns a float array aligned with the frame rows.
    """
    feats = params["features"]
    X = frame[feats].to_numpy(float)
    ok = ~np.isnan(X).any(axis=1)
    p = np.full(len(frame), np.nan)
    if ok.any():
        z = logit_scores(params["logit"], standardize(X[ok], params["scaler"]))
        p[ok] = apply_platt(z, params["calibration"]) if params.get("calibration") else sigmoid(z)
    return p
