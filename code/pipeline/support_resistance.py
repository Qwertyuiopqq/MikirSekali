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
                                   H >= config.SR_HURST_TRENDING "trending"
                                   (levels break more easily),
                                   H <= config.SR_HURST_MEAN_REVERTING
                                   "mean_reverting" (levels tend to hold),
                                   otherwise "random". "unknown" if no Hurst
                                   column.

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
    sr_break_up, sr_hit_support                         (events, see below)

Events (`add_sr_events`) -- what the fuzzy system's signed breakout component needs
------------------------------------------------------------------------------------
By construction Close always sits BETWEEN the two lines of its own row, so "price broke
resistance" can only be seen against the lines as they were YESTERDAY (known at yesterday's
close -> no look-ahead):
    broke resistance : today's Close  > yesterday's resistance line
    hit support      : today's Low   <= yesterday's support line   (a wick that reaches it counts)
Only levels that were touched >= config.SR_SIGNAL_MIN_TOUCHES times, or the 1-year extreme
(touches == 0 fallback), count -- the nearest weak line is ~3-4% away and "hit" it on a quarter of
all days. Each event then fades linearly over config.SR_EVENT_PERSIST_DAYS trading days
(strength 1.0 -> 0.0); a break of resistance is cancelled early if Close falls back below the
broken level. Both columns are strengths in 0..1 (NaN until lines exist).
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
SR_EVENT_COLUMNS = ["sr_break_up", "sr_hit_support"]
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


def _event_strengths(close, low, sup, res, sup_touch, res_touch, min_touches, tol, margin, persist):
    """
    Per-trading-day event strengths (0..1) for ONE symbol; all inputs are 1-D arrays over its
    trading days, oldest first. The lines at index t-1 are what was known when day t opened.
    """
    n = len(close)
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    age_up = age_dn = None
    broken = np.nan                      # resistance level that was broken
    for t in range(1, n):
        s_prev, r_prev = sup[t - 1], res[t - 1]
        if not (np.isfinite(s_prev) and np.isfinite(r_prev)):
            continue                     # no lines yet -> unknown, leave NaN
        # touches == 0 is the 1-year high/low fallback: the strongest level there is, so it counts too
        res_ok = res_touch[t - 1] >= min_touches or res_touch[t - 1] == 0
        sup_ok = sup_touch[t - 1] >= min_touches or sup_touch[t - 1] == 0

        if res_ok and close[t] > r_prev * (1.0 + margin):
            age_up, broken = 0, r_prev
        elif age_up is not None:
            age_up += 1
            if age_up >= persist or close[t] <= broken:   # faded out, or the break failed
                age_up = None

        if sup_ok and low[t] <= s_prev * (1.0 + tol):
            age_dn = 0
        elif age_dn is not None:
            age_dn += 1
            if age_dn >= persist:
                age_dn = None

        up[t] = 0.0 if age_up is None else 1.0 - age_up / persist
        dn[t] = 0.0 if age_dn is None else 1.0 - age_dn / persist
    return up, dn


def add_sr_events(df: pd.DataFrame, min_touches=None, tol=None, margin=None, persist=None) -> pd.DataFrame:
    """
    Adds `sr_break_up` and `sr_hit_support` (strengths 0..1, see module docstring) computed from the
    S/R line columns already in `df` (Close/Low/Volume/symbol/Date + the four sr_* line columns).
    Returns a copy with the SAME index and row order as `df`; if the line columns are missing both
    new columns are all-NaN (the fuzzy breakout component then simply has no signal).
    Works on a finished MCS_health.csv too -- the website server uses that to cope with older files.
    """
    min_touches = config.SR_SIGNAL_MIN_TOUCHES if min_touches is None else min_touches
    tol = config.SR_TOUCH_TOL if tol is None else tol
    margin = config.SR_BREAK_MARGIN if margin is None else margin
    persist = max(1, int(config.SR_EVENT_PERSIST_DAYS if persist is None else persist))

    out = df.copy()
    n = len(out)
    up_all = np.full(n, np.nan)
    dn_all = np.full(n, np.nan)
    need = {"Close", "sr_support", "sr_resistance", "sr_support_touches", "sr_resistance_touches"}
    if n and need.issubset(out.columns):
        codes = pd.factorize(out["symbol"], sort=True)[0] if "symbol" in out.columns else np.zeros(n, dtype=int)
        dates = (pd.to_datetime(out["Date"]).to_numpy("datetime64[ns]").astype("int64")
                 if "Date" in out.columns else np.arange(n))
        vol = out["Volume"].to_numpy(dtype=float) if "Volume" in out.columns else np.ones(n)
        close = out["Close"].to_numpy(dtype=float)
        low = out["Low"].fillna(out["Close"]).to_numpy(dtype=float) if "Low" in out.columns else close
        sup, res = out["sr_support"].to_numpy(float), out["sr_resistance"].to_numpy(float)
        sup_t, res_t = out["sr_support_touches"].to_numpy(float), out["sr_resistance_touches"].to_numpy(float)

        order = np.lexsort((dates, codes))                     # by symbol, then date
        cuts = np.flatnonzero(np.diff(codes[order])) + 1
        for gpos in np.split(order, cuts):                     # row positions of one symbol, oldest first
            trad = gpos[vol[gpos] > 0]                         # events are judged on trading days only
            if trad.size < 2:
                continue
            u, d = _event_strengths(close[trad], low[trad], sup[trad], res[trad],
                                    sup_t[trad], res_t[trad], min_touches, tol, margin, persist)
            # weekend/holiday rows inherit the latest trading day (same convention as the lines)
            up_all[gpos] = pd.Series(u, index=trad).reindex(gpos).ffill().to_numpy()
            dn_all[gpos] = pd.Series(d, index=trad).reindex(gpos).ffill().to_numpy()

    out["sr_break_up"] = up_all
    out["sr_hit_support"] = dn_all
    return out


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

    df = add_sr_events(df)
    ev = df[SR_EVENT_COLUMNS].notna().all(axis=1)
    print(f"[support_resistance] Event S/R: 'broke resistance' aktif di {int((df['sr_break_up'] > 0).sum())} baris, "
          f"'hit support' aktif di {int((df['sr_hit_support'] > 0).sum())} baris (dari {int(ev.sum())} baris berevent).")
    return df
