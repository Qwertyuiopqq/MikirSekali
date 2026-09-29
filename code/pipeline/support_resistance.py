"""
support_resistance.py
----------------------
Optional pipeline stage (5c): per-row support / resistance LINES, drawn
next to the per-company LSTM-Hurst breakout probability.

How the pieces fit together
---------------------------
  * LINES  (this module)        -> swing-pivot clustering, pure numpy/pandas.
  * PROBABILITY (LSTM-Hurst)    -> `breakout_probability`, one model per
                                   company (lstm_hurst_predictor.py): the
                                   chance that Close exceeds the yearly
                                   resistance (`yearly_resistance`) in the
                                   next 30 days.
  * REGIME (Hurst exponent)     -> `sr_regime`, read from the
                                   `hurst_exponent` column that
                                   lstm_hurst_predictor.py already computed:
                                   H >= 0.55 "trending" (levels break more
                                   easily), H <= 0.45 "mean_reverting"
                                   (levels tend to hold), otherwise
                                   "random". "unknown" if no Hurst column.

So every row ends up with: the two nearest lines around today's Close,
how many times each was touched (strength), the market regime, and the
LSTM's breakout probability for the yearly high.

Line method (light + causal)
----------------------------
  1. Work on TRADING days only (Volume > 0). The calendar spine forward-
     fills weekends/holidays into flat plateaus, which would break swing
     detection and double-count touches.
  2. Swing pivots: a High is a pivot high if it is the maximum of the
     `2k+1` bars centred on it (k = SR_PIVOT_WINDOW); likewise Low for
     pivot lows. A pivot only becomes usable k bars AFTER it happened
     (that is when it is confirmed) -> no look-ahead bias at any row.
  3. For each day, take the pivots confirmed within the last
     SR_LOOKBACK_TRADING_DAYS trading days, sort their prices and merge
     neighbours closer than SR_CLUSTER_ATR_MULT x ATR into one level
     (level = mean price, touches = number of pivots merged). Highs and
     lows are clustered TOGETHER, so an old resistance that price
     already crossed naturally becomes a support (polarity flip).
  4. support = closest level below Close, resistance = closest level
     above, preferring levels touched at least SR_MIN_TOUCHES (2) times.
     If a side has no such level the closest weaker one is used (touches
     = 1); if it has no level at all it falls back to the 1-year low /
     high with touches = 0 ("extreme, not a confirmed cluster").

Cost: one small numpy sort per trading day per symbol (a few seconds for
10 symbols x 5 years), no extra dependencies.

Public entry point: `add_support_resistance(df, paths=None) -> pd.DataFrame`
Adds columns (rows are never dropped, NaN while there is too little history):
    sr_support, sr_resistance
    sr_support_touches, sr_resistance_touches
    sr_dist_to_support_pct, sr_dist_to_resistance_pct   (fraction of Close)
    sr_regime
"""
import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:  # running as a plain script, not as a package
    import config

SR_COLUMNS = [
    "sr_support", "sr_resistance",
    "sr_support_touches", "sr_resistance_touches",
    "sr_dist_to_support_pct", "sr_dist_to_resistance_pct",
    "sr_regime",
]
MIN_HISTORY_TRADING_DAYS = 60  # same minimum as yearly_* in the fractal features


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    return pd.Series(tr).rolling(period, min_periods=1).mean().to_numpy()


def _pivot_flags(values: np.ndarray, k: int, kind: str) -> np.ndarray:
    """True where values[i] is the max (kind='high') / min ('low') of the 2k+1 window around i."""
    s = pd.Series(values)
    roll = s.rolling(2 * k + 1, center=True, min_periods=2 * k + 1)
    ref = roll.max() if kind == "high" else roll.min()
    flags = (s == ref).to_numpy().copy()  # copy: pandas>=3 returns read-only arrays
    # A flat top/bottom (equal values on neighbouring bars) would count as several
    # pivots for the same touch -> keep only the first bar of each tie.
    idx = np.flatnonzero(flags)
    for i in idx:
        lo = max(0, i - k)
        if flags[lo:i].any() and np.any(values[lo:i][flags[lo:i]] == values[i]):
            flags[i] = False
    return flags


def _cluster(prices: np.ndarray, tol: float):
    """Anchored 1-D clustering of sorted prices. Returns (levels, touches)."""
    if prices.size == 0:
        return np.empty(0), np.empty(0, dtype=int)
    p = np.sort(prices)
    levels, touches = [], []
    start = 0
    for i in range(1, p.size):
        if p[i] - p[start] > tol:
            levels.append(p[start:i].mean())
            touches.append(i - start)
            start = i
    levels.append(p[start:].mean())
    touches.append(p.size - start)
    return np.asarray(levels), np.asarray(touches, dtype=int)


def _lines_for_trading_days(high, low, close, k, lookback, atr_mult, atr_period):
    """Vectorised-per-day core. Inputs are 1-D float arrays over TRADING days only."""
    n = len(close)
    out = np.full((n, 4), np.nan)  # support, resistance, sup_touches, res_touches
    if n < max(MIN_HISTORY_TRADING_DAYS, 2 * k + 2):
        return out

    atr = _atr(high, low, close, atr_period)
    hi_idx = np.flatnonzero(_pivot_flags(high, k, "high"))
    lo_idx = np.flatnonzero(_pivot_flags(low, k, "low"))
    piv_idx = np.concatenate([hi_idx, lo_idx])
    piv_px = np.concatenate([high[hi_idx], low[lo_idx]])
    order = np.argsort(piv_idx, kind="stable")
    piv_idx, piv_px = piv_idx[order], piv_px[order]
    piv_conf = piv_idx + k  # first day the pivot is known -> no look-ahead

    for t in range(MIN_HISTORY_TRADING_DAYS - 1, n):
        first = np.searchsorted(piv_idx, t - lookback, side="right")   # pivot happened within lookback
        last = np.searchsorted(piv_conf, t, side="right")              # pivot already confirmed at t
        levels, touches = _cluster(piv_px[first:last], atr_mult * atr[t])

        w0 = max(0, t - lookback + 1)
        px = close[t]

        # Prefer CONFIRMED levels (>= SR_MIN_TOUCHES pivots merged); if a side has none,
        # settle for the closest weaker level; if it has no level at all, use the 1-year extreme.
        strong = touches >= config.SR_MIN_TOUCHES
        for col, side_mask in ((0, levels < px), (1, levels > px)):
            pick = side_mask & strong
            if not pick.any():
                pick = side_mask
            if pick.any():
                cand = np.flatnonzero(pick)
                j = cand[-1] if col == 0 else cand[0]   # levels ascending: closest below = last, closest above = first
                out[t, col], out[t, col + 2] = levels[j], touches[j]
            else:
                out[t, col] = low[w0:t + 1].min() if col == 0 else high[w0:t + 1].max()
                out[t, col + 2] = 0
    return out


def _regime_from_hurst(h: pd.Series) -> pd.Series:
    regime = np.select(
        [h >= config.SR_HURST_TRENDING, h <= config.SR_HURST_MEAN_REVERTING, h.notna()],
        ["trending", "mean_reverting", "random"],
        default="unknown",
    )
    return pd.Series(regime, index=h.index)


def add_support_resistance(df: pd.DataFrame, paths=None, hurst_col: str = "hurst_exponent") -> pd.DataFrame:
    """
    Adds the `SR_COLUMNS` to `df` (per symbol, causal, no rows dropped).
    `paths` is accepted for symmetry with the other stages and is unused.
    Never raises on missing optional columns: without High/Low it works on
    Close, without `hurst_col` the regime is "unknown".
    """
    df = df.sort_values(["symbol", "Date"]).reset_index(drop=True)
    for c in SR_COLUMNS:
        df[c] = np.nan
    df["sr_regime"] = "unknown"

    have_hl = {"High", "Low"}.issubset(df.columns)
    if not have_hl:
        print("[support_resistance] Kolom High/Low tidak ada -> pivot dihitung dari Close.")

    for sym, g in df.groupby("symbol"):
        trading = g[g["Volume"] > 0] if "Volume" in g.columns else g
        if trading.empty:
            continue
        close = trading["Close"].to_numpy(dtype=float)
        high = trading["High"].fillna(trading["Close"]).to_numpy(dtype=float) if have_hl else close
        low = trading["Low"].fillna(trading["Close"]).to_numpy(dtype=float) if have_hl else close

        res = _lines_for_trading_days(
            high, low, close,
            k=config.SR_PIVOT_WINDOW, lookback=config.SR_LOOKBACK_TRADING_DAYS,
            atr_mult=config.SR_CLUSTER_ATR_MULT, atr_period=config.SR_ATR_PERIOD,
        )
        tmp = pd.DataFrame(res, index=trading.index,
                           columns=["sr_support", "sr_resistance", "sr_support_touches", "sr_resistance_touches"])
        # non-trading days (weekend/holiday) inherit the latest trading-day lines;
        # Close on those days is a forward-fill, so this is identical to recomputing.
        tmp = tmp.reindex(g.index).ffill()
        df.loc[g.index, tmp.columns] = tmp

    close = df["Close"].astype(float)
    df["sr_dist_to_support_pct"] = (df["sr_support"] - close) / close      # <= 0 (line is below price)
    df["sr_dist_to_resistance_pct"] = (df["sr_resistance"] - close) / close  # >= 0 (line is above price)

    if hurst_col in df.columns:
        df["sr_regime"] = _regime_from_hurst(df[hurst_col])

    n_ok = int(df["sr_support"].notna().sum())
    print(f"[support_resistance] Garis S/R terisi untuk {n_ok}/{len(df)} baris "
          f"({df['symbol'].nunique()} simbol, pivot window={config.SR_PIVOT_WINDOW}, "
          f"lookback={config.SR_LOOKBACK_TRADING_DAYS} hari trading).")
    return df
