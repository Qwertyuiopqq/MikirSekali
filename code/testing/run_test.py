"""
run_test.py
------------
CLI entry point for the testing/ toolkit: point it at a new CSV (in
the MCS_features master-calendar-spine format) and it will run the
already-trained XGBoost models on it, score MAE/MAPE/RMSE against any
ground truth in the file, and compute both health_score_real and
health_score_predict.

Usage
-----
    python run_test.py --input path/to/new_data.csv

    # override any pipeline/config.py Paths field the same way as pipeline.py:
    python run_test.py --input path/to/new_data.csv \\
        --xgboost-models-folder ../models/XGBoost \\
        --output-folder ../data/test_output

Outputs (written to --output-folder, default ../../data/output)
-----------------------------------------------------------------
    MCS_test_health.csv   input + XGBoost prediction + health_score_real + health_score_predict
    MCS_test_metrics.csv  per-symbol + overall MAE / MAPE / RMSE (only if the
                           input CSV had target_volume_T_plus_1 to compare against)
"""
import argparse
import dataclasses
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from pipeline import config

try:
    from .test_runner import run_test
except ImportError:  # running as `python run_test.py`, not as a package
    from test_runner import run_test


def parse_args(paths: config.Paths):
    parser = argparse.ArgumentParser(
        description="Test the MCS pipeline's XGBoost models + health score on a new CSV."
    )
    parser.add_argument("--input", required=True,
                         help="Path to the new CSV. Must follow the MCS_features spine format "
                              "(Date, symbol, all XGBOOST_FEATURES columns; "
                              "target_volume_T_plus_1 is optional -- include it for MAE/MAPE).")
    for f in dataclasses.fields(paths):
        parser.add_argument(f"--{f.name.replace('_', '-')}", default=None)
    args = parser.parse_args()

    overrides = {k: v for k, v in vars(args).items() if k != "input" and v is not None}
    return args.input, dataclasses.replace(paths, **overrides)


def main():
    input_csv, paths = parse_args(config.Paths())
    try:
        run_test(input_csv, paths)
    except FileNotFoundError as e:
        print(f"\nERROR: file tidak ditemukan - {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\nERROR: format input tidak valid - {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
