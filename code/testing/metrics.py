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


def root_mean_squared_error(actual, predicted) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def build_metrics_table(df: pd.DataFrame, actual_col: str, predicted_col: str,
                         group_col: str = "symbol") -> pd.DataFrame:
    """
    Per-symbol MAE / MAPE / RMSE, plus a pooled 'ALL' row.

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
            "RMSE": root_mean_squared_error(clean[actual_col], clean[predicted_col]),
        })

    table = pd.DataFrame(rows)
    if not table.empty:
        table[["MAE", "MAPE_%", "RMSE"]] = table[["MAE", "MAPE_%", "RMSE"]].round(4)
    return table
