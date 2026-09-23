"""
pipeline.py
-----------
Orchestrates the full Master Calendar Spine (MCS) pipeline end to end
and writes out the four deliverables:

    MCS_raw.csv      Stage 1: calendar-complete raw spine
    MCS_report.csv   Stage 2: MCS_raw + enrichment + REAL health score
    MCS_predict.csv  Stage 3: MCS_raw + enrichment + XGBoost prediction
    MCS_health.csv   Stage 4: MCS_predict + PREDICTED health score

Usage
-----
    python -m mcs_pipeline.pipeline
    python -m mcs_pipeline.pipeline --history-folder ../../dataset/history5y \
                                     --xgboost-models-folder ../XGBoost \
                                     --output-folder ./output

Any Paths field in config.py can be overridden with a matching --flag
(dashes instead of underscores).
"""
import argparse
import dataclasses
import sys

try:
    from . import config
    from .spine_builder import build_raw_spine
    from .enrichment import enrich_features
    from .xgboost_predictor import run_predictions
    from .fuzzy_system import BusinessHealthFuzzySystem
except ImportError:  # running as `python pipeline.py`, not as a package
    import config
    from spine_builder import build_raw_spine
    from enrichment import enrich_features
    from xgboost_predictor import run_predictions
    from fuzzy_system import BusinessHealthFuzzySystem


def parse_args(paths: config.Paths) -> config.Paths:
    parser = argparse.ArgumentParser(description="Run the MCS pipeline.")
    for f in dataclasses.fields(paths):
        parser.add_argument(f"--{f.name.replace('_', '-')}", default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if v is not None}
    return dataclasses.replace(paths, **overrides)


def run_pipeline(paths: config.Paths = None) -> dict:
    """
    Execute all four stages. Returns a dict of {name: DataFrame} and also
    writes each stage to disk as it completes.
    """
    paths = paths or config.Paths()
    fuzzy_sys = BusinessHealthFuzzySystem()
    results = {}

    # ---- Stage 1: raw spine --------------------------------------------
    print("\n=== STAGE 1/4: Membangun Master Calendar Spine (RAW) ===")
    df_raw = build_raw_spine(paths)
    raw_out = paths.output_path(paths.mcs_raw_csv)
    df_raw.to_csv(raw_out, index=False)
    print(f"✅ MCS_raw disimpan di: {raw_out}  ({len(df_raw)} baris)")
    results["MCS_raw"] = df_raw

    # ---- Stage 2: enrichment (shared by report & predict) --------------
    print("\n=== Menyuntikkan fitur enrichment (sentiment/macro/fundamental) ===")
    df_enriched = enrich_features(df_raw, paths)

    # ---- Stage 2a: REAL health score -> MCS_report ----------------------
    print("\n=== STAGE 2/4: Menghitung Health Score REAL -> MCS_report ===")
    df_report = df_enriched.copy()
    df_report["health_score_real"] = fuzzy_sys.calculate_health_score(df_report, mode="real")
    report_out = paths.output_path(paths.mcs_report_csv)
    df_report.to_csv(report_out, index=False)
    print(f"✅ MCS_report disimpan di: {report_out}  ({len(df_report)} baris)")
    results["MCS_report"] = df_report

    # ---- Stage 3: XGBoost prediction -> MCS_predict ---------------------
    print("\n=== STAGE 3/4: Menjalankan prediksi XGBoost -> MCS_predict ===")
    df_predict = run_predictions(df_enriched, paths)
    predict_out = paths.output_path(paths.mcs_predict_csv)
    df_predict.to_csv(predict_out, index=False)
    print(f"✅ MCS_predict disimpan di: {predict_out}  ({len(df_predict)} baris)")
    results["MCS_predict"] = df_predict

    # ---- Stage 4: PREDICTED health score -> MCS_health -------------------
    print("\n=== STAGE 4/4: Menghitung Health Score PREDICT -> MCS_health ===")
    df_health = df_predict.copy()
    df_health["health_score_predict"] = fuzzy_sys.calculate_health_score(
        df_health, mode="predict", xgb_col=config.PREDICTION_COLUMN
    )
    health_out = paths.output_path(paths.mcs_health_csv)
    df_health.to_csv(health_out, index=False)
    print(f"✅ MCS_health disimpan di: {health_out}  ({len(df_health)} baris)")
    results["MCS_health"] = df_health

    print("\n🎉 Pipeline selesai. 4 file berhasil dibuat:")
    for name, fname in [
        ("MCS_raw", paths.mcs_raw_csv), ("MCS_report", paths.mcs_report_csv),
        ("MCS_predict", paths.mcs_predict_csv), ("MCS_health", paths.mcs_health_csv),
    ]:
        print(f"   - {name}: {paths.output_path(fname)}")

    return results


def main():
    paths = parse_args(config.Paths())
    try:
        run_pipeline(paths)
    except FileNotFoundError as e:
        print(f"\nERROR: file tidak ditemukan - {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: terjadi kesalahan sistem - {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
