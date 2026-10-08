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

    TRAINING -- run these from this folder (or upload MCS_features.csv and
              the .py files they import to Colab):
                python train_xgboost_transfer.py   -> <config.xgboost_models_folder>
                          (base + per-company models + xgb_training_meta.json)
                python lstm_hurst_train.py         -> <config.lstm_hurst_model_path>
                          (_POOLED/ model; --mode both adds per-company ones;
                           --no-lstm trains the logistic model only, no torch)
              Both print a hold-out report; try `--smoke` first (1 minute).

    LOCAL  -- python pipeline.py score
      Stage 5: run the downloaded XGBoost models on
               MCS_features.csv                              -> MCS_predict.csv
      Stage 5b: breakout probability + Hurst exponent (optional)
      Stage 5c: support/resistance lines + Hurst regime (optional)
      Stage 6: fuzzy health scores (REAL + PREDICTED, both
               recomputed here with the current formulas)    -> MCS_health.csv
               The file is written to `Paths.mcs_health_csv_path`
               in config.py -- the same place website/server.py reads.

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
from pathlib import Path

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
def _print_scoring_summary(df_health: pd.DataFrame, fuzzy_sys: BusinessHealthFuzzySystem, has_breakout: bool):
    """Shows which fuzzy components actually carried signal, so nothing is silently scored as noise."""
    mode = "predict_breakout" if has_breakout else "predict"
    memb = fuzzy_sys.memberships(df_health, mode, config.PREDICTION_COLUMN,
                                 "breakout_probability" if has_breakout else None)
    n = len(memb)
    print(f"   Komponen yang membawa sinyal (dari {n} baris; sisanya dikeluarkan & bobot dinormalisasi ulang):")
    for c in memb.columns:
        print(f"     - {c:<10}: {int(memb[c].notna().sum()):>6}/{n} baris")
    if "sentiment_is_mock" in df_health.columns and df_health["sentiment_is_mock"].astype(bool).all():
        print("   ⚠️ Sentimen di dataset ini MOCK (acak) karena file sentimen asli tidak ada -> komponen sentimen "
              "TIDAK dihitung dalam skor.\n"
              f"      Sediakan {config.Paths().sentiment_scores_csv} (mis. via `python website/sentiment_trigger.py "
              "--save-csv`) lalu jalankan `prepare` + `score` lagi.")
    if not has_breakout:
        print("   -> breakout_probability tidak tersedia (belum ada model breakout yang lolos uji hold-out / riwayat kurang) -> komponen breakout tidak dihitung.")


def run_score(paths: config.Paths) -> dict:
    """Stages 5-6: XGBoost prediction -> combined real+predicted health table."""
    fuzzy_sys = BusinessHealthFuzzySystem()

    features_path = paths.output_path(paths.mcs_features_csv)
    try:
        df_features = pd.read_csv(features_path, parse_dates=["Date"])
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{features_path} tidak ditemukan. Jalankan `python pipeline.py prepare` dulu."
        )

    print("\n=== STAGE 5/6: Menjalankan prediksi XGBoost -> MCS_predict ===")
    df_predict = run_predictions(df_features, paths)

    print("\n=== STAGE 5b/6: (opsional) Probabilitas breakout + Hurst (model gabungan atau per perusahaan) ===")
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

    print("\n=== STAGE 6/6: Menghitung skor REAL + PREDICTED (fuzzy) -> MCS_health ===")
    # One call scores both columns with the SAME code website/server.py uses, and adds the
    # helper columns (xgb_applicable, xgb_norm, sr_break_up, sr_hit_support) so every number
    # on the website can be traced back to a column in this file.
    if "sentiment_is_mock" not in df_predict.columns:
        # MCS_features.csv written before enrichment.py flagged its sentiment: it is the random mock
        # unless the real sentiment file exists (same rule website/server.py applies to an old CSV).
        df_predict["sentiment_is_mock"] = not Path(paths.resolve(paths.sentiment_scores_csv)).exists()
    df_health = fuzzy_sys.score_frame(df_predict)
    _print_scoring_summary(df_health, fuzzy_sys, has_breakout)

    health_out = Path(paths.health_csv())          # location is set in config.py (mcs_health_csv_path)
    health_out.parent.mkdir(parents=True, exist_ok=True)
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
