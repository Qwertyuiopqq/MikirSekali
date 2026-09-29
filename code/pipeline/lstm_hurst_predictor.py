"""
lstm_hurst_predictor.py
------------------------
Optional pipeline stage: fractal (Hurst exponent + annual support/
resistance) features, plus a "breakout probability" score from a
pre-trained LSTMWithHurst model, fed into the fuzzy health system as an
extra variable alongside daily_news_sentiment and the XGBoost volume
prediction.

ONE MODEL PER COMPANY. Each symbol is scored by its own LSTM (own weights,
own StandardScaler, own hidden_size), stored in its own sub-folder:

    <paths.lstm_hurst_model_path>/
        ASSA/  best_lstm_hurst_model.pth
               lstm_hurst_scaler.joblib
               lstm_hurst_best_params.json
        BIRD/  ...                       (folder name = symbol without ".JK")

A symbol whose folder or any of the files is missing is skipped with a
warning; its breakout_probability stays NaN (the fuzzy stage then scores
that row without the breakout component, see pipeline.py). There is
deliberately NO cross-company fallback: a model trained on company A's
price/volatility range produces meaningless scores for company B.

Mirrors enrichment.py's philosophy: every external dependency (the
`hurst` package, the saved .pth weights, torch itself) is optional and
wrapped so a missing piece degrades gracefully -- the rest of the
pipeline (XGBoost volume + real-mode fuzzy score) must keep working even
if this stage can't run.

Public entry point: `add_breakout_probability(df, paths) -> pd.DataFrame`
Adds a `breakout_probability` column (float in [0, 1], NaN where it
couldn't be computed) without dropping any rows. If at least one company
model is available it also leaves the fractal columns (yearly_resistance,
yearly_support, dist_to_resistance, dist_to_support, hurst_exponent) in
the frame, which support_resistance.py uses for its Hurst regime label.

--------------------------------------------------------------------
NOTES
1. Inference needs the EXACT same StandardScaler (mean/std) used at
   training time, or the model sees a different distribution than it
   learned on and its output is meaningless. Each company folder must
   therefore contain its own lstm_hurst_scaler.joblib; without it that
   company is skipped rather than silently refit.

2. lstm_hurst_best_params.json holds hidden_size (and seq_length, metrics)
   per company. If it is missing (old checkpoint), this module tries every
   hidden_size in the tuning grid (64, then 32) as a workaround.

3. What the probability means: P(Close exceeds `yearly_resistance`
   within the next 30 days), the label used in lstm_hurst_prepare.py.
   `yearly_resistance` is a 252-row rolling max of Close on the calendar
   spine (weekends included), same definition as training.

4. Hurst window = config.HURST_WINDOW (100 rows). The `hurst` package raises
   on series shorter than 100 points, so the earlier window of 60 always fell
   into the `return 0.5` fallback (a constant feature). Models trained with
   the old window must be retrained -- their scaler/weights saw a constant 0.5.
--------------------------------------------------------------------
"""
import os

import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:  # running as a plain script, not as a package
    import config

FRACTAL_FEATURES = ["Close", "dist_to_resistance", "dist_to_support", "hurst_exponent"]
SEQ_LENGTH = 30
HIDDEN_SIZE_CANDIDATES = [64, 32]  # matches param_grid['hidden_size'] in the tuning script


def compute_fractal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds yearly_resistance, yearly_support, dist_to_resistance,
    dist_to_support and hurst_exponent, per symbol. Same definitions as
    the pasted training script so inference-time features match what the
    model was trained on.

    Requires the `hurst` package (`pip install hurst`); if it's missing,
    hurst_exponent is filled with 0.5 (the script's own "random walk"
    fallback value) for every row instead of crashing the pipeline.
    """
    df = df.sort_values(["symbol", "Date"]).reset_index(drop=True).copy()

    try:
        from hurst import compute_Hc
        hurst_available = True
    except ImportError:
        print("[lstm_hurst] Paket 'hurst' tidak terpasang -> hurst_exponent diisi 0.5 "
              "(default random-walk) untuk semua baris. Jalankan `pip install hurst` "
              "agar fitur ini dihitung sungguhan.")
        hurst_available = False

    def _calc_hurst(series):
        if not hurst_available:
            return 0.5
        try:
            H, _, _ = compute_Hc(series, kind="price", simplified=True)
            return H
        except Exception:
            return 0.5

    def _per_symbol(group):
        group = group.copy()
        group["yearly_resistance"] = group["Close"].rolling(window=252, min_periods=60).max()
        group["yearly_support"] = group["Close"].rolling(window=252, min_periods=60).min()
        group["dist_to_resistance"] = (group["yearly_resistance"] - group["Close"]) / group["Close"]
        group["dist_to_support"] = (group["Close"] - group["yearly_support"]) / group["Close"]
        group["hurst_exponent"] = group["Close"].rolling(window=config.HURST_WINDOW).apply(_calc_hurst, raw=False)
        return group

    # NOTE: deliberately NOT `df.groupby('symbol').apply(_per_symbol)`.
    # pandas >=2.2 changed groupby(...).apply(...) to exclude the grouping
    # column from what the function receives (and, from pandas 3.0, from
    # the result too) -- the exact pattern used in the original pasted
    # script. Depending on the pandas version in your Colab runtime that
    # either silently drops `symbol` from the output (breaking every
    # `df['symbol']` used afterwards) or raises a KeyError. An explicit
    # loop + concat sidesteps the ambiguity entirely and behaves the same
    # on every pandas version.
    parts = [_per_symbol(g) for _, g in df.groupby("symbol")]
    out = pd.concat(parts, ignore_index=True)
    if hurst_available and out["hurst_exponent"].dropna().nunique() <= 1:
        print("[lstm_hurst] ⚠️ hurst_exponent KONSTAN untuk semua baris -> compute_Hc gagal terus dan "
              f"jatuh ke fallback 0.5. Paket 'hurst' butuh window >= 100 (config.HURST_WINDOW={config.HURST_WINDOW}).")
    return out


def _build_model(input_size: int, hidden_size: int):
    import torch.nn as nn

    class LSTMWithHurst(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
            self.dropout = nn.Dropout(0.3)
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_step = lstm_out[:, -1, :]
            out = self.dropout(last_step)
            out = self.fc(out)
            return self.sigmoid(out)

    return LSTMWithHurst()


def _read_params(company_dir: str) -> dict:
    import json
    params_path = os.path.join(company_dir, config.LSTM_PARAMS_FILENAME)
    if os.path.exists(params_path):
        with open(params_path) as f:
            return json.load(f)
    return {}


def _load_model(model_path: str, params: dict, tag: str = ""):
    """
    Uses hidden_size from the company's best_params.json when present.
    Falls back to trying each candidate hidden_size only if it isn't
    there (e.g. a checkpoint saved by an older training script).
    """
    import torch

    state_dict = torch.load(model_path, map_location="cpu")

    if "hidden_size" in params:
        hidden_size = int(params["hidden_size"])
        model = _build_model(input_size=len(FRACTAL_FEATURES), hidden_size=hidden_size)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    print(f"[lstm_hurst] {tag}: hidden_size tidak ada di params -> menebak dari {HIDDEN_SIZE_CANDIDATES}.")
    last_err = None
    for hidden_size in HIDDEN_SIZE_CANDIDATES:
        model = _build_model(input_size=len(FRACTAL_FEATURES), hidden_size=hidden_size)
        try:
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except RuntimeError as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Checkpoint tidak cocok dengan hidden_size manapun di {HIDDEN_SIZE_CANDIDATES}. "
        f"Simpan best_params saat training. Error terakhir: {last_err}"
    )


def _company_name(symbol: str) -> str:
    return symbol.replace(".JK", "")


def _company_dir(paths, symbol: str, models_folder: str = None) -> str:
    """<models_folder>/<COMPANY>/ (models_folder defaults to paths.lstm_hurst_model_path)."""
    root = models_folder or getattr(paths, "lstm_hurst_model_path", "../../models/LSTMwithHurst")
    return os.path.join(root, _company_name(symbol))


def _missing_files(company_dir: str) -> list:
    needed = [config.LSTM_MODEL_FILENAME, config.LSTM_SCALER_FILENAME]
    return [f for f in needed if not os.path.exists(os.path.join(company_dir, f))]


def add_breakout_probability(df: pd.DataFrame, paths, models_folder: str = None,
                              output_col: str = "breakout_probability") -> pd.DataFrame:
    """
    Adds `output_col` (probability in [0, 1], NaN where unavailable), scoring
    every symbol with ITS OWN LSTM-Hurst model.
    Never drops rows and never raises -- if torch or a company's model/
    scaler aren't available, prints a warning and leaves that company's
    output_col NaN, exactly like enrichment.py's fallback behaviour for
    optional data sources.

    models_folder overrides paths.lstm_hurst_model_path (the folder that holds
    one sub-folder per company).
    """
    df = df.copy()
    df[output_col] = np.nan

    try:
        import torch
    except ImportError:
        print("[lstm_hurst] Paket 'torch' tidak terpasang -> melewati breakout_probability.")
        return df

    root = models_folder or getattr(paths, "lstm_hurst_model_path", "../../models/LSTMwithHurst")
    if not os.path.isdir(root):
        print(f"[lstm_hurst] Folder model LSTM-Hurst tidak ada ({root}) -> melewati breakout_probability.")
        return df

    symbols = list(df["symbol"].unique())
    available = {}
    for sym in symbols:
        cdir = _company_dir(paths, sym, models_folder)
        missing = _missing_files(cdir)
        if missing:
            print(f"  [-] [lstm_hurst] {sym}: model tidak lengkap di {cdir} "
                  f"(tidak ada: {', '.join(missing)}) -> breakout_probability kosong untuk {sym}.")
        else:
            available[sym] = cdir

    if not available:
        print(f"[lstm_hurst] Tidak ada satupun model perusahaan yang lengkap di {root} -> melewati breakout_probability.")
        return df

    import joblib

    df = compute_fractal_features(df)
    df[output_col] = np.nan  # compute_fractal_features re-sorted/rebuilt the frame
    feat_ready = df.dropna(subset=FRACTAL_FEATURES)

    pred_rows = []
    for sym, cdir in available.items():
        g = feat_ready[feat_ready["symbol"] == sym].sort_values("Date")
        params = _read_params(cdir)
        seq_len = int(params.get("seq_length", SEQ_LENGTH))
        if len(g) < seq_len:
            print(f"  [-] [lstm_hurst] {sym}: histori {len(g)} hari < {seq_len} -> dilewati.")
            continue
        try:
            scaler = joblib.load(os.path.join(cdir, config.LSTM_SCALER_FILENAME))
            model = _load_model(os.path.join(cdir, config.LSTM_MODEL_FILENAME), params, tag=sym)
        except Exception as e:
            print(f"  [-] [lstm_hurst] {sym}: gagal memuat model/scaler ({e}) -> dilewati.")
            continue

        X_scaled = scaler.transform(g[FRACTAL_FEATURES])
        seqs = np.stack([X_scaled[i:i + seq_len] for i in range(len(g) - seq_len + 1)])
        with torch.no_grad():
            probs = model(torch.tensor(seqs, dtype=torch.float32)).numpy().ravel()
        # prediction i corresponds to the LAST day of window i, i.e. row (seq_len - 1 + i)
        dates = g["Date"].iloc[seq_len - 1:].values
        pred_rows.append(pd.DataFrame({"symbol": sym, "Date": dates, output_col: probs}))
        print(f"  [+] [lstm_hurst] {sym}: {len(probs)} skor (hidden_size={getattr(model.lstm, 'hidden_size', '?')}, seq_length={seq_len}).")

    if pred_rows:
        # pd.merge (not a python dict keyed on (symbol, Date) tuples) so a
        # Timestamp-vs-datetime64 dtype mismatch between the two sides
        # can't silently fail to match -- pandas aligns the join key by
        # value, not by Python hash.
        preds_df = pd.concat(pred_rows, ignore_index=True)
        df = df.drop(columns=[output_col]).merge(preds_df, on=["symbol", "Date"], how="left")
        n_scored = df[output_col].notna().sum()
        print(f"[lstm_hurst] breakout_probability terisi untuk {n_scored}/{len(df)} baris "
              f"({len(pred_rows)}/{len(symbols)} simbol punya model LSTM-Hurst sendiri).")
    else:
        print("[lstm_hurst] Tidak ada simbol yang berhasil diskor -> breakout_probability kosong.")

    return df
