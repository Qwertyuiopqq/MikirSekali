"""
lstm_hurst_predictor.py
------------------------
Stage 5b of the pipeline (optional): adds, for every company,

    yearly_resistance, yearly_support, dist_to_resistance, dist_to_support, hurst_exponent   (always)
    breakout_probability                                                                      (when a usable model exists)
    breakout_model                                  which model produced it, e.g. "pooled-logit" / "ASSA-lstm"

breakout_probability = P(Close exceeds its 252-SESSION high within the next 21 sessions); see lstm_features.py for the exact
definitions. Everything is computed on trading-day rows and carried forward onto weekends/holidays (the same convention the
support/resistance lines use). The model comes from lstm_hurst_train.py:

    <lstm_hurst_model_path>/_POOLED/            one model for every company (the default)
    <lstm_hurst_model_path>/<COMPANY>/          optional per-company model, overrides the pooled one

A company with no usable model gets NaN (the fuzzy score then simply leaves the breakout component out) -- the pipeline
never fails because of this stage. A model is NOT usable when it was
  * trained by the OLD trainer (no feature_version: raw Close level, calendar-row windows) -> retrain;
  * trained with different window settings than config.py now has;
  * flagged `beats_baseline: false` in its own hold-out report (config.LSTM_USE_ONLY_IF_BEATS_BASELINE).
A model trained as an LSTM but run where torch is missing falls back to the logistic model stored next to it.

Public entry point: `add_breakout_probability(df, paths, models_folder=None, output_col="breakout_probability")`
"""
import os

import numpy as np
import pandas as pd

try:
    from . import config
    from . import lstm_features as lf
except ImportError:  # running as a plain script, not as a package
    import config
    import lstm_features as lf

_WINDOW_KEYS = {"seq_length": "LSTM_SEQ_LENGTH", "horizon": "LSTM_HORIZON", "yearly_window": "LSTM_YEARLY_WINDOW",
                "yearly_min_periods": "LSTM_YEARLY_MIN_PERIODS", "vol_window": "LSTM_VOL_WINDOW", "hurst_window": "HURST_WINDOW"}


def _group_status(model_dir):
    """(params or None, reason it is unusable or None). `params` is None and `reason` None when there is no folder at all."""
    params = lf.read_params(model_dir)
    if not os.path.isdir(model_dir) or not params:
        return None, ("legacy" if os.path.isdir(model_dir) else None)
    if params.get("feature_version") != config.LSTM_FEATURE_VERSION:
        return None, "legacy"
    off = [k for k, c in _WINDOW_KEYS.items() if params.get(k) != getattr(config, c)]
    if off:
        return None, "windows differ from config.py (" + ", ".join(f"{k}: model {params.get(k)} vs config {getattr(config, _WINDOW_KEYS[k])}" for k in off) + ") -> retrain"
    if config.LSTM_USE_ONLY_IF_BEATS_BASELINE and params.get("beats_baseline") is False:
        return None, "its own hold-out report says it did not beat the baselines (config.LSTM_USE_ONLY_IF_BEATS_BASELINE)"
    return params, None


def _lstm_probabilities(params, model_dir, frame):
    """Probabilities of the stored LSTM for the frame's rows, or None when torch / the weights are not available."""
    path = os.path.join(model_dir, config.LSTM_MODEL_FILENAME)
    if not os.path.exists(path):
        return None
    try:
        try:
            from . import lstm_model as lm
        except ImportError:
            import lstm_model as lm
    except ImportError:
        return None
    smp = lf.build_samples(frame, params["features"], params["seq_length"], with_label=False)
    p = np.full(len(frame), np.nan)
    if len(smp["end"]):
        X = lf.standardize(smp["X"], params["scaler"]).astype(np.float32)
        model = lm.load_lstm(path, X.shape[2], params["lstm"]["hidden_size"], params["lstm"].get("dropout", 0.3))
        p[smp["end"]] = lf.apply_platt(lm.predict_logits(model, X), params["lstm"]["calibration"])
    return p


def add_breakout_probability(df: pd.DataFrame, paths, models_folder: str = None,
                             output_col: str = "breakout_probability") -> pd.DataFrame:
    root = models_folder or paths.lstm_hurst_model_path
    df = df.copy()
    df[output_col] = np.nan
    df["breakout_model"] = pd.Series([None] * len(df), index=df.index, dtype="object")

    if lf.get_hurst_estimator() is None:
        print("[lstm_hurst_predictor] ⚠️ paket `hurst` tidak terpasang (pip install hurst): hurst_exponent dan "
              "breakout_probability tidak bisa dihitung (kolom S/R tetap dibuat).")
    frames = lf.build_symbol_frames(df, with_label=False)
    df = lf.export_fractal_columns(df, frames)                       # the S/R regime needs hurst_exponent even without a model

    pooled, pooled_why = _group_status(os.path.join(root, config.LSTM_POOLED_DIRNAME))
    legacy_dirs, skipped = [], []
    if pooled_why == "legacy":
        legacy_dirs.append(config.LSTM_POOLED_DIRNAME)
    elif pooled_why:
        skipped.append(f"{config.LSTM_POOLED_DIRNAME}: {pooled_why}")

    done = 0
    symbols = sorted(frames)
    print(f"[lstm_hurst_predictor] {len(symbols)} simbol; model gabungan ({config.LSTM_POOLED_DIRNAME}): "
          f"{pooled['model_type'] if pooled else 'tidak ada'}")
    for sym in symbols:
        short = sym.replace(".JK", "")
        own_dir = os.path.join(root, short)
        own, why = _group_status(own_dir)
        if why == "legacy":
            legacy_dirs.append(short)
        elif why:
            skipped.append(f"{short}: {why}")
        params, model_dir, group = (own, own_dir, short) if own else (pooled, os.path.join(root, config.LSTM_POOLED_DIRNAME), "pooled")
        if params is None:
            print(f"  [-] {sym}: tidak ada model breakout yang bisa dipakai -> breakout_probability = NaN")
            continue
        fr = frames[sym]
        label = f"{group}-{params['model_type']}"
        p = None
        if params["model_type"] == "lstm":
            p = _lstm_probabilities(params, model_dir, fr)
            if p is None:
                print(f"  [i] {sym}: model LSTM tidak bisa dijalankan (torch/berkas bobot tidak ada) -> memakai model logistik yang tersimpan")
                label += "(fallback-logit)"
        if p is None:
            p = lf.logit_probabilities(params, fr)
        ok = ~np.isnan(p)
        if not ok.any():
            why_not = ("fitur Hurst tidak tersedia (paket `hurst` belum terpasang)" if lf.get_hurst_estimator() is None
                       else f"riwayat belum cukup untuk fitur model ({len(fr)} sesi; perlu >= {config.HURST_WINDOW + config.LSTM_YEARLY_MIN_PERIODS // 2})")
            print(f"  [-] {sym}: {why_not} -> NaN")
            continue
        idx = df.index[df["symbol"] == sym]
        cal = lf.calendar_ffill(df.loc[idx, "Date"], pd.DataFrame({"Date": fr["Date"], output_col: p}))
        df.loc[idx, output_col] = cal[output_col].to_numpy()
        df.loc[idx[cal[output_col].notna().to_numpy()], "breakout_model"] = label
        print(f"  [+] {sym}: {label}, {int(ok.sum())} sesi berprobabilitas")
        done += 1

    if legacy_dirs:
        print(f"[lstm_hurst_predictor] ⚠️ {len(legacy_dirs)} folder model LAMA dilewati ({', '.join(legacy_dirs)}): dilatih dengan metode lama "
              "(fitur Close mentah, jendela hari kalender, evaluasi pada test set). Latih ulang: `python lstm_hurst_train.py`.")
    for s in skipped:
        print(f"[lstm_hurst_predictor] model dilewati -> {s}")
    if not done:
        print("[lstm_hurst_predictor] Tidak ada probabilitas breakout (komponen breakout dikeluarkan dari skor). "
              "Untuk mengaktifkan: `python lstm_hurst_train.py` lalu `python pipeline.py score`.")
    return df
