"""
pipeline.py
-----------
Two-phase orchestrator, because training now happens externally in
Google Colab between two local runs.

    LOCAL  -- python pipeline.py prepare
      Stage 1: build the raw calendar spine                -> MCS_raw.csv
      Stage 2: enrich it (real or mock sentiment, macro,
               fundamentals) into XGBoost-ready features    -> MCS_features.csv
      Stage 3: fuzzy health score on REAL data only          -> MCS_report.csv

    MANUAL -- upload MCS_features.csv to Colab, run
              train_xgboost_transfer.py, download the
              base + per-company models into
              <config.xgboost_models_folder> (default models/XGBoost/)
              Optional: lstm_hurst_prepare.py + lstm_hurst_tune_train.py
              (Colab) -> unzip LSTMwithHurst.zip into models/ so that
              models/LSTMwithHurst/<COMPANY>/ holds each company's LSTM.

    LOCAL  -- python pipeline.py score
      Stage 5: run the downloaded XGBoost models on
               MCS_features.csv                              -> MCS_predict.csv
      Stage 5b: per-company LSTM-Hurst breakout probability (optional)
      Stage 5c: support/resistance lines + Hurst regime (optional)
      Stage 6: fuzzy health score on PREDICTED data, merged
               side-by-side with the REAL score computed in
               Stage 3                                        -> MCS_health.csv

    python pipeline.py all   runs prepare then score back-to-back --
    only useful once the models already exist locally (e.g. re-running
    scoring after retraining, or in a test/dev environment).

Usage
-----
    python pipeline.py prepare
    python pipeline.py score
    python pipeline.py all

Any Paths field in config.py can be overridden with a matching --flag
(dashes instead of underscores), e.g. --history-folder, --output-folder.
"""
import argparse
import dataclasses
import sys

try:
    from . import config
    from .spine_builder import build_raw_spine
    from .enrichment import enrich_features
    from .xgboost_predictor import run_predictions
    from .lstm_hurst_predictor import add_breakout_probability
    from .support_resistance import add_support_resistance
    from .fuzzy_system import BusinessHealthFuzzySystem
except ImportError:  # running as `python pipeline.py`, not as a package
    import config
    from spine_builder import build_raw_spine
    from enrichment import enrich_features
    from xgboost_predictor import run_predictions
    from lstm_hurst_predictor import add_breakout_probability
    from support_resistance import add_support_resistance
    from fuzzy_system import BusinessHealthFuzzySystem

import numpy as np
import pandas as pd


def parse_args(paths: config.Paths):
    parser = argparse.ArgumentParser(description="Run the MCS pipeline.")
    parser.add_argument("stage", choices=["prepare", "score", "all"], default="all", nargs="?",
                         help="'prepare' (stages 1-3, run before Colab training), "
                              "'score' (stages 5-6, run after downloading trained models), "
                              "or 'all' (both, for local dev/testing).")
    for f in dataclasses.fields(paths):
        parser.add_argument(f"--{f.name.replace('_', '-')}", default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k != "stage" and v is not None}
    return args.stage, dataclasses.replace(paths, **overrides)


# ---------------------------------------------------------------------------
# Phase 1: LOCAL, before Colab training
# ---------------------------------------------------------------------------
def run_prepare(paths: config.Paths) -> dict:
    """Stages 1-3: raw spine -> features -> real health score."""
    fuzzy_sys = BusinessHealthFuzzySystem()

    print("\n=== STAGE 1/6: Membangun Master Calendar Spine (RAW) ===")
    df_raw = build_raw_spine(paths)
    raw_out = paths.output_path(paths.mcs_raw_csv)
    df_raw.to_csv(raw_out, index=False)
    print(f"✅ MCS_raw disimpan di: {raw_out}  ({len(df_raw)} baris)")

    print("\n=== STAGE 2/6: Preprocessing fitur untuk XGBoost (MCS_features) ===")
    df_features = enrich_features(df_raw, paths)
    features_out = paths.output_path(paths.mcs_features_csv)
    df_features.to_csv(features_out, index=False)
    print(f"✅ MCS_features disimpan di: {features_out}  ({len(df_features)} baris)")
    print(f"   -> Unggah file ini ke Colab untuk training/fine-tuning XGBoost.")

    print("\n=== STAGE 3/6: Menghitung Health Score REAL -> MCS_report ===")
    df_report = df_features.copy()
    df_report["health_score_real"] = fuzzy_sys.calculate_health_score(df_report, mode="real")
    report_out = paths.output_path(paths.mcs_report_csv)
    df_report.to_csv(report_out, index=False)
    print(f"✅ MCS_report disimpan di: {report_out}  ({len(df_report)} baris)")

    print("\n⏸  Jalankan train_finetune_xgboost.py di Google Colab dengan MCS_features.csv,")
    print(f"   lalu taruh semua model (.json) hasil download ke folder: {paths.xgboost_models_folder}/")
    print("   Setelah itu jalankan: python pipeline.py score")

    return {"MCS_raw": df_raw, "MCS_features": df_features, "MCS_report": df_report}


# ---------------------------------------------------------------------------
# Phase 2: LOCAL, after downloading trained models from Colab
# ---------------------------------------------------------------------------
def run_score(paths: config.Paths) -> dict:
    """Stages 5-6: XGBoost prediction -> combined real+predicted health table."""
    fuzzy_sys = BusinessHealthFuzzySystem()

    features_path = paths.output_path(paths.mcs_features_csv)
    report_path = paths.output_path(paths.mcs_report_csv)
    try:
        df_features = pd.read_csv(features_path, parse_dates=["Date"])
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{features_path} tidak ditemukan. Jalankan `python pipeline.py prepare` dulu."
        )

    print("\n=== STAGE 5/6: Menjalankan prediksi XGBoost -> MCS_predict ===")
    df_predict = run_predictions(df_features, paths)

    print("\n=== STAGE 5b/6: (opsional) Skor breakout LSTM-Hurst (1 model per perusahaan) ===")
    df_predict = add_breakout_probability(df_predict, paths)
    has_breakout = "breakout_probability" in df_predict.columns and df_predict["breakout_probability"].notna().any()

    print("\n=== STAGE 5c/6: (opsional) Garis support/resistance (pivot ringan + regime Hurst) ===")
    try:
        df_predict = add_support_resistance(df_predict, paths)
    except Exception as e:  # optional stage: never block the rest of the pipeline
        print(f"   ⚠️ Garis support/resistance dilewati: {e}")

    predict_out = paths.output_path(paths.mcs_predict_csv)
    df_predict.to_csv(predict_out, index=False)
    print(f"✅ MCS_predict disimpan di: {predict_out}  ({len(df_predict)} baris)")

    print("\n=== STAGE 6/6: Menggabungkan skor REAL + PREDICTED -> MCS_health ===")
    df_health = df_predict.copy()
    if has_breakout:
        score_breakout = fuzzy_sys.calculate_health_score(
            df_health, mode="predict_breakout",
            xgb_col=config.PREDICTION_COLUMN, breakout_col="breakout_probability",
        )
        score_plain = fuzzy_sys.calculate_health_score(
            df_health, mode="predict", xgb_col=config.PREDICTION_COLUMN
        )
        # Row-level fallback: with one LSTM per company, some companies (or the first
        # ~90 days of each) can lack a breakout_probability. Scoring those rows with the
        # breakout weight would treat "no signal" as "0% breakout" and drag them down,
        # so they get the plain 'predict' formula instead.
        has_bp = df_health["breakout_probability"].notna().to_numpy()
        df_health["health_score_predict"] = np.where(has_bp, score_breakout, score_plain)
        print(f"   -> breakout_probability dipakai di {int(has_bp.sum())}/{len(has_bp)} baris "
              f"(mode='predict_breakout'); sisanya memakai mode='predict' (tanpa komponen breakout).")
    else:
        df_health["health_score_predict"] = fuzzy_sys.calculate_health_score(
            df_health, mode="predict", xgb_col=config.PREDICTION_COLUMN
        )
        print("   -> breakout_probability tidak tersedia (belum ada model LSTM-Hurst per perusahaan) "
              "-> health_score_predict dihitung tanpa komponen breakout (mode='predict', seperti sebelumnya).")

    # Bring health_score_real back in (computed in Stage 3) so the final
    # table lets you compare actual vs. predicted health side-by-side.
    try:
        df_report = pd.read_csv(report_path, parse_dates=["Date"])
        real_scores = df_report[["Date", "symbol", "health_score_real"]]
        df_health = pd.merge(df_health, real_scores, on=["Date", "symbol"], how="left")
    except FileNotFoundError:
        print(f"   ⚠️ {report_path} tidak ditemukan, MCS_health hanya akan berisi health_score_predict.")

    health_out = paths.output_path(paths.mcs_health_csv)
    df_health.to_csv(health_out, index=False)
    print(f"✅ MCS_health disimpan di: {health_out}  ({len(df_health)} baris)")

    return {"MCS_predict": df_predict, "MCS_health": df_health}


def run_pipeline(paths: config.Paths = None, stage: str = "all") -> dict:
    paths = paths or config.Paths()
    results = {}
    if stage in ("prepare", "all"):
        results.update(run_prepare(paths))
    if stage in ("score", "all"):
        results.update(run_score(paths))
    return results


def main():
    stage, paths = parse_args(config.Paths())
    try:
        run_pipeline(paths, stage=stage)
    except FileNotFoundError as e:
        print(f"\nERROR: file tidak ditemukan - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: terjadi kesalahan sistem - {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
