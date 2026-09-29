"""
metrics.py
----------
Accuracy metrics for comparing XGBoost volume predictions against
actual next-day volume (target_volume_T_plus_1), plus a helper to
build a per-symbol + overall summary table.

Pure logic, no I/O -- easy to unit-test in isolation, same philosophy
as fuzzy_system.py in the main pipeline.
"""
import numpy as np
import pandas as pd


def mean_absolute_error(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual - predicted)))


def mean_absolute_percentage_error(actual, predicted, drop_zero_actual: bool = True):
    """
    Returns (mape_percent, n_excluded).

    MAPE divides by the actual value, so rows where actual == 0 are
    undefined (div-by-zero). Those rows are excluded from the MAPE
    calculation by default (they still count toward MAE). n_excluded
    tells you how many rows that was, so a low-volume symbol doesn't
    silently produce a misleading MAPE.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = actual != 0 if drop_zero_actual else np.ones_like(actual, dtype=bool)
    n_excluded = int((~mask).sum())

    if mask.sum() == 0:
        return float("nan"), n_excluded

    pct_err = np.abs((actual[mask] - predicted[mask]) / actual[mask])
    return float(np.mean(pct_err) * 100), n_excluded


def weighted_mean_absolute_percentage_error(actual, predicted) -> float:
    """
    WMAPE = sum(|actual - predicted|) / sum(|actual|) * 100.

    Unlike plain MAPE, this divides by the TOTAL actual volume instead of
    dividing row-by-row, so a single near-zero-volume day can no longer
    blow up the whole metric -- it just contributes its (small) share of
    both the numerator and the denominator. No rows need to be excluded.
    Standard choice for intermittent/low-liquidity series (e.g. BLOG.JK),
    where plain MAPE is dominated by a handful of thin-volume days.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(actual - predicted)) / denom * 100)


def symmetric_mean_absolute_percentage_error(actual, predicted) -> float:
    """
    SMAPE = mean( |actual - predicted| / ((|actual| + |predicted|) / 2) ) * 100.

    Bounded (0-200%) and symmetric between over- and under-prediction.
    Rows where both actual and predicted are exactly 0 are treated as a
    perfect match (0% error) instead of being excluded or causing a
    divide-by-zero, since "predicted no volume on a truly no-volume day"
    is a correct prediction, not an undefined one.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = (np.abs(actual) + np.abs(predicted)) / 2.0
    safe_denom = np.where(denom == 0, 1.0, denom)
    pct_err = np.where(denom == 0, 0.0, np.abs(actual - predicted) / safe_denom)
    return float(np.mean(pct_err) * 100)


def root_mean_squared_error(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def build_metrics_table(df: pd.DataFrame, actual_col: str, predicted_col: str,
                         group_col: str = "symbol") -> pd.DataFrame:
    """
    Per-symbol MAE / MAPE / WMAPE / SMAPE / RMSE, plus a pooled 'ALL' row.

    MAPE_% keeps the old (fragile) definition for continuity/comparison,
    with rows where actual == 0 excluded (see MAPE_rows_excluded_zero_actual).
    WMAPE_% and SMAPE_% don't need that exclusion and are the more
    reliable numbers to actually judge a symbol by, especially thin/
    intermittent-volume ones -- prefer those over MAPE_% when they disagree.

    Rows with NaN in either actual_col or predicted_col (e.g. a symbol
    with no matching model, or the last-known day with no real future
    value yet) are dropped before scoring.
    """
    clean = df.dropna(subset=[actual_col, predicted_col]).copy()

    rows = []
    for sym, g in clean.groupby(group_col):
        mape, n_ex = mean_absolute_percentage_error(g[actual_col], g[predicted_col])
        rows.append({
            group_col: sym,
            "n_rows": len(g),
            "MAE": mean_absolute_error(g[actual_col], g[predicted_col]),
            "MAPE_%": mape,
            "MAPE_rows_excluded_zero_actual": n_ex,
            "WMAPE_%": weighted_mean_absolute_percentage_error(g[actual_col], g[predicted_col]),
            "SMAPE_%": symmetric_mean_absolute_percentage_error(g[actual_col], g[predicted_col]),
            "RMSE": root_mean_squared_error(g[actual_col], g[predicted_col]),
        })

    if not clean.empty:
        mape_all, n_ex_all = mean_absolute_percentage_error(clean[actual_col], clean[predicted_col])
        rows.append({
            group_col: "ALL",
            "n_rows": len(clean),
            "MAE": mean_absolute_error(clean[actual_col], clean[predicted_col]),
            "MAPE_%": mape_all,
            "MAPE_rows_excluded_zero_actual": n_ex_all,
            "WMAPE_%": weighted_mean_absolute_percentage_error(clean[actual_col], clean[predicted_col]),
            "SMAPE_%": symmetric_mean_absolute_percentage_error(clean[actual_col], clean[predicted_col]),
            "RMSE": root_mean_squared_error(clean[actual_col], clean[predicted_col]),
        })

    table = pd.DataFrame(rows)
    if not table.empty:
        metric_cols = ["MAE", "MAPE_%", "WMAPE_%", "SMAPE_%", "RMSE"]
        table[metric_cols] = table[metric_cols].round(4)
    return table


def build_metrics_report(df: pd.DataFrame, actual_col: str, predicted_col: str,
                          group_col: str = "symbol") -> dict:
    """
    Two metrics tables side by side:

      - 'all_days': every row with both actual + predicted present. On a
        calendar spine this INCLUDES non-trading days (weekends,
        holidays), where the real market didn't trade and the actual
        next-day volume is 0 by construction. Those rows are already
        excluded from MAPE (division by zero), but a model that
        predicts nonzero volume on them still gets penalized in MAE and
        RMSE here -- so 'all_days' MAE/RMSE can look inflated for
        reasons that have nothing to do with forecasting skill.
      - 'trading_days_only': the same rows, but restricted to
        actual_col != 0 (i.e. only days the market actually traded).
        This is usually the more meaningful number to judge the model
        by, since it isn't diluted by non-trading zero-volume days.

    Both tables use the same MAE/MAPE/WMAPE/SMAPE/RMSE definitions from
    this module. For thin/illiquid symbols (frequent near-zero actual
    volume, e.g. BLOG.JK), prefer WMAPE_%/SMAPE_% over MAPE_% -- MAPE_%
    can still explode on non-zero-but-small actual values even though
    exact zeros are excluded.
    """
    all_days = build_metrics_table(df, actual_col, predicted_col, group_col)

    trading_df = df[df[actual_col] != 0]
    trading_days_only = build_metrics_table(trading_df, actual_col, predicted_col, group_col)

    return {"all_days": all_days, "trading_days_only": trading_days_only}
