"""
test_runner.py
----------------
Core orchestration for the testing/ toolkit. Given a new CSV that
follows the MCS_features master-calendar-spine format, this:

  1. Validates it against that schema (input_validator.py).
  2. health_score_real   -- fuzzy health score on the data AS GIVEN
     ("real data" only). Same formula as pipeline.py Stage 3.
  3. Runs the EXISTING per-symbol XGBoost models on it (reuses
     pipeline.xgboost_predictor.run_predictions -- no duplicated
     model-loading/prediction logic to drift out of sync).
  4. health_score_predict -- fuzzy health score on REAL + PREDICTED
     ("real + future data"). Same formula as pipeline.py Stage 6.
  5. If the input has ground-truth target_volume_T_plus_1, scores the
     predictions with MAE / MAPE / RMSE (metrics.py), per symbol and
     pooled overall. If it doesn't (pure future/forecast input), this
     step is skipped and you still get predictions + both health
     scores.

Public entry point: `run_test(input_csv_path, paths=None) -> dict`
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from pipeline import config
from pipeline.xgboost_predictor import run_predictions
from pipeline.fuzzy_system import BusinessHealthFuzzySystem

try:
    from . import metrics as metrics_mod
    from .input_validator import validate_spine_format
except ImportError:  # running as a plain script, not as a package
    import metrics as metrics_mod
    from input_validator import validate_spine_format


def load_test_csv(input_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    return df


def run_test(input_csv_path: str, paths: config.Paths = None,
             feature_cols=None, save_outputs: bool = True) -> dict:
    paths = paths or config.Paths()
    feature_cols = feature_cols or config.XGBOOST_FEATURES

    print(f"[test_runner] Membaca file uji: {input_csv_path}")
    df_input = load_test_csv(input_csv_path)

    validation = validate_spine_format(df_input, feature_cols=feature_cols)
    for w in validation.warnings:
        print(f"   ⚠️ {w}")
    validation.raise_if_invalid()
    print("   ✅ Format input sesuai master calendar spine.")

    fuzzy_sys = BusinessHealthFuzzySystem()

    # --- 1) health score on REAL data only -----------------------------
    print("\n=== Health score (REAL data only) ===")
    df_real = df_input.copy()
    df_real["health_score_real"] = fuzzy_sys.calculate_health_score(df_real, mode="real")
    print(f"   ✅ health_score_real dihitung untuk {len(df_real)} baris.")

    # --- 2) predictions using the EXISTING per-symbol XGBoost models --
    print("\n=== Menjalankan model XGBoost yang sudah dilatih ===")
    df_pred = run_predictions(df_input, paths, feature_cols=feature_cols)
    if df_pred.empty:
        print("   ⚠️ Tidak ada baris dengan prediksi valid -- cek folder model "
              f"({paths.xgboost_models_folder}) atau kelengkapan fitur input.")

    # --- 3) health score on REAL + PREDICTED (future) data -------------
    print("\n=== Health score (REAL + FUTURE/PREDICTED data) ===")
    df_pred["health_score_predict"] = fuzzy_sys.calculate_health_score(
        df_pred, mode="predict", xgb_col=config.PREDICTION_COLUMN
    )
    print(f"   ✅ health_score_predict dihitung untuk {len(df_pred)} baris.")

    df_health = pd.merge(
        df_pred, df_real[["Date", "symbol", "health_score_real"]],
        on=["Date", "symbol"], how="left",
    )

    # --- 4) accuracy metrics (only if ground truth is present) ---------
    metrics_table = None
    if validation.has_target:
        print("\n=== Menghitung MAE / MAPE / RMSE (prediksi vs. aktual) ===")
        metrics_table = metrics_mod.build_metrics_table(
            df_health, actual_col=config.TARGET_COLUMN,
            predicted_col=config.PREDICTION_COLUMN,
        )
        if metrics_table.empty:
            print("   ⚠️ Tidak ada baris dengan target aktual DAN prediksi -- metrik dilewati.")
            metrics_table = None
        else:
            print(metrics_table.to_string(index=False))
    else:
        print("\n=== MAE / MAPE dilewati (tidak ada target_volume_T_plus_1 di input) ===")

    results = {
        "MCS_test_real": df_real,       # health_score_real only, no prediction step
        "MCS_test_predict": df_pred,    # + XGBoost prediction + health_score_predict
        "MCS_test_health": df_health,   # predict table + health_score_real merged back in
        "metrics": metrics_table,       # None if no ground truth was available
    }

    if save_outputs:
        health_out = paths.output_path("MCS_test_health.csv")
        df_health.to_csv(health_out, index=False)
        print(f"\n✅ MCS_test_health disimpan di: {health_out} ({len(df_health)} baris)")

        if metrics_table is not None:
            metrics_out = paths.output_path("MCS_test_metrics.csv")
            metrics_table.to_csv(metrics_out, index=False)
            print(f"✅ MCS_test_metrics disimpan di: {metrics_out}")

    return results
