"""
xgboost_predictor.py
----------------------
Stage 3 of the pipeline.

Loads one fine-tuned XGBoost model per symbol (JSON format) and predicts
next-day volume for that symbol's rows, writing the result into a new
prediction column.

Model set (produced by train_xgboost_transfer.py): 1 base model
(base_transport_model.json, pooled across every symbol) + 1 fine-tuned
model per company (finetuned_<COMPANY>_model.json). Each symbol uses its
OWN fine-tuned model. Only if that file is missing does it fall back to
the base model (with a warning, since the base model has not seen that
company specifically). If neither exists the symbol is skipped with a
warning rather than raising.

Public entry point: `run_predictions(df, paths, feature_cols) -> pd.DataFrame`
Produces the data behind MCS_predict.csv (before health scoring).
"""
import os

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from . import config
except ImportError:  # running as a plain script, not as a package
    import config


def _model_path_for_symbol(paths, symbol: str) -> str:
    company_name = symbol.replace(".JK", "")
    filename = paths.xgboost_model_filename_tpl.format(company=company_name)
    return os.path.join(paths.xgboost_models_folder, filename)


def _base_model_path(paths) -> str:
    filename = getattr(paths, "xgboost_base_model_filename", "base_transport_model.json")
    return os.path.join(paths.xgboost_models_folder, filename)


def run_predictions(df: pd.DataFrame, paths, feature_cols=None,
                     prediction_col: str = config.PREDICTION_COLUMN) -> pd.DataFrame:
    """
    Predict next-day volume per symbol using per-symbol XGBoost models.

    Parameters
    ----------
    df : pd.DataFrame            enriched frame (must contain feature_cols + 'symbol')
    paths : mcs_pipeline.config.Paths
    feature_cols : list[str]     defaults to config.XGBOOST_FEATURES
    prediction_col : str         name of the output column

    Returns
    -------
    pd.DataFrame, a copy of df with `prediction_col` added, restricted to
    rows where required technical features and the prediction are present.
    """
    df = df.copy()
    feature_cols = feature_cols or config.XGBOOST_FEATURES
    df[prediction_col] = np.nan

    symbols = df["symbol"].unique()
    print(f"[xgboost_predictor] Ditemukan {len(symbols)} simbol. Memulai prediksi...")

    for sym in symbols:
        mask = df["symbol"] == sym
        X_infer = df.loc[mask, feature_cols]

        model_path = _model_path_for_symbol(paths, sym)
        used_base = False
        if not os.path.exists(model_path):
            base_path = _base_model_path(paths)
            if os.path.exists(base_path):
                print(f"  [~] WARNING: model fine-tuned untuk {sym} tidak ditemukan ({model_path}) "
                      f"-> memakai BASE model ({base_path}).")
                model_path = base_path
                used_base = True
            else:
                print(f"  [-] WARNING: model tidak ditemukan untuk {sym} ({model_path}), "
                      f"base model juga tidak ada ({base_path})")
                continue

        model = xgb.XGBRegressor()
        model.load_model(model_path)
        # Model dilatih dengan target np.log1p(volume) (lihat
        # train_xgboost_transfer.py) supaya satu base model bisa menangani
        # simbol dengan skala volume yang berbeda ~1000x. Wajib di-inverse
        # dengan expm1 di sini, kalau tidak hasilnya akan berupa angka
        # skala-log (mis. ~14-20) dan bukan volume asli (jutaan lembar).
        raw_pred = model.predict(X_infer)
        df.loc[mask, prediction_col] = np.clip(np.expm1(raw_pred), a_min=0, a_max=None)
        print(f"  [+] Prediksi sukses untuk {sym}" + (" (base model)" if used_base else ""))

    required_cols = ["MA_7_Close", "MA_30_Close", "volatility_7d", prediction_col]
    df = df.dropna(subset=required_cols)
    return df.reset_index(drop=True)
