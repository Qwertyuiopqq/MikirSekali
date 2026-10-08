#!/usr/bin/env python3
"""
train_xgboost_transfer.py
--------------------------
Trains the XGBoost "next-session volume" models: one pooled BASE model across every company plus, where it
actually helps, one fine-tuned model per company (transfer learning, `xgb_model=<base>`).

Run it from the pipeline/ folder (needs config.py next to it), after `python pipeline.py prepare`:

    python train_xgboost_transfer.py                    # full run
    python train_xgboost_transfer.py --smoke            # ~1 minute sanity check (2 companies, few rounds, no refit)
    python train_xgboost_transfer.py --symbols GIAA,SMDR --no-refit

Writes (names are the ones xgboost_predictor.py already reads):
    <models>/base_transport_model.json
    <models>/finetuned_<COMPANY>_model.json            (only for companies where the fine-tune won on validation)
    <models>/xgb_training_meta.json                     (target definition, features, split dates, hold-out report)
    <models>/xgb_holdout_report.csv

WHAT CHANGED versus the old trainer, and why (measured on the 10-company dataset)
  1. TARGET = volume of the next TRADING SESSION (spine_builder.py builds it), not tomorrow's calendar day.
     The calendar-day target was 0 on 37% of the rows (28.5% weekends + 8.2% weekday holidays/no-trade days), which
     pulled every forecast down ~2x (median actual/forecast 1.94) and made Fri/Sat forecasts ~0. Removing the
     zeros cut the held-out WMAPE from 75% to 55% and lifted the rise/fall correlation from 0.40 to 0.56.
  2. HOLD-OUT EVALUATION. Rows are split chronologically with ONE set of date cut-offs for all companies (so no
     company's future leaks into another's training), with an embargo between the blocks:
         train (first 60% of the dates) | embargo | validation (next 20%) | embargo | test (last 20%)
     The number of boosting rounds is chosen on validation (early stopping), the fine-tune is kept per company only
     if it beats the base model on validation, and the TEST block is only ever reported, never used to choose.
     The old script scored the model on its own training rows ("how well it memorised").
  3. NAIVE BASELINES are reported next to the model ("volume = today", "mean of the last 5 sessions"). Even
     in-sample the old forecast was worse than the 5-session mean (WMAPE 65.9% vs 62.5%): if a model does not beat
     those, it adds nothing.
  4. The rise/fall correlation is reported too, because that (forecast vs the model's own same-weekday norm) is
     what the fuzzy health score actually uses.
  5. Imputation matches inference: idx_macro_close is only forward-filled (the old bfill looked into the future and
     inference never had it); NaN returns/volatility stay NaN (XGBoost handles NaN natively, as at inference).
  6. Final models are REFIT on all rows (unless --no-refit) with the rounds picked above. The reported metrics are
     those of the model before the refit, i.e. an estimate for the deployed one.

The model is still trained on log1p(volume) and xgboost_predictor.py inverts it with expm1.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from . import config
except ImportError:  # running as a plain script, not as a package
    import config

FEATURES = list(config.XGBOOST_FEATURES)
TARGET = config.TARGET_COLUMN

# ---- settings (edit here) -----------------------------------------------------------------------------------------
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2          # test = the remaining 20% of the dates
EMBARGO_DAYS = 7                         # calendar days dropped between blocks (the target looks <= a few sessions ahead)
BASE_PARAMS = {"objective": "reg:squarederror", "eval_metric": "mae", "learning_rate": 0.05,
               "max_depth": 6, "subsample": 0.8, "seed": config.RANDOM_SEED}
TUNE_PARAMS = {"objective": "reg:squarederror", "eval_metric": "mae", "learning_rate": 0.01,
               "max_depth": 4, "subsample": 0.8, "seed": config.RANDOM_SEED}
ROUNDS_MAX, PATIENCE = 600, 40           # base model: at most this many rounds, stop after PATIENCE without improvement
TUNE_ROUNDS_MAX, TUNE_PATIENCE = 150, 20
MIN_TRAIN_ROWS, MIN_VAL_ROWS = 150, 40   # a company with fewer rows in a block keeps the base model
MIN_IMPROVEMENT = 0.01                   # fine-tune must cut the validation WMAPE by >= 1% (relative) to be kept


# ---- metrics ------------------------------------------------------------------------------------------------------
def wmape(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return 100.0 * np.abs(a - p).sum() / np.abs(a).sum()


def smape(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return 100.0 * np.mean(np.abs(a - p) / ((np.abs(a) + np.abs(p)) / 2.0 + 1e-9))


def level_metrics(actual, pred):
    """Accuracy of the forecast LEVEL. bias = median(actual / forecast): 1.0 means unbiased, 2.0 = forecasts 2x too low."""
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    return {"n": int(len(a)), "WMAPE_%": wmape(a, p), "SMAPE_%": smape(a, p),
            "MAE": float(np.abs(a - p).mean()), "bias": float(np.median(a / np.maximum(p, 1.0)))}


def rise_fall_corr(d, pred):
    """
    Correlation between the predicted and the realised CHANGE in volume, each measured against its own same-weekday
    norm (the construction fuzzy_system.py uses for the health score). Returns (model, naive "today's volume") or NaNs.
    """
    try:
        try:
            from .fuzzy_system import forecast_norm
        except ImportError:
            from fuzzy_system import forecast_norm
    except Exception:
        return np.nan, np.nan
    x = d[["Date", "symbol", "day_of_week"]].copy()
    x["pred"], x["act"], x["vol"] = np.asarray(pred, float), d[TARGET].to_numpy(float), d["Volume"].to_numpy(float)
    ok = np.ones(len(x), dtype=bool)
    n_fc, n_ac, n_vol = (forecast_norm(x, c, ok) for c in ("pred", "act", "vol"))
    m = (n_fc.notna() & n_ac.notna() & n_vol.notna()).to_numpy()
    if m.sum() < 30:
        return np.nan, np.nan
    real = np.log(x["act"] / n_ac)[m]
    model = np.log(np.maximum(x["pred"], 1.0) / np.maximum(n_fc, 1.0))[m]
    naive = np.log(np.maximum(x["vol"], 1.0) / np.maximum(n_vol, 1.0))[m]
    return float(model.corr(real)), float(naive.corr(real))


# ---- data ---------------------------------------------------------------------------------------------------------
def load_frame(csv_path, symbols=None):
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    missing = [c for c in FEATURES + [TARGET, "symbol", "Date", "Volume", "day_of_week"] if c not in df.columns]
    if missing:
        sys.exit(f"[!] {csv_path} lacks columns: {missing}. Run `python pipeline.py prepare` first.")
    df = df.sort_values(["symbol", "Date"]).reset_index(drop=True)
    if config.XGB_TARGET_MODE == "next_trading_day" and (df[TARGET] == 0).mean() > 0.001:
        sys.exit(f"[!] {csv_path} still has the OLD target (volume of tomorrow's CALENDAR day: {(df[TARGET] == 0).mean():.0%} zeros).\n"
                 f"    Re-run `python pipeline.py prepare` so {TARGET} becomes the next trading session's volume, then train again.")

    # naive baseline: mean of the last 5 sessions' volume (causal), looked up per row
    tr = df[df["Volume"] > 0]
    avg5 = tr.groupby("symbol")["Volume"].transform(lambda s: s.rolling(5, min_periods=3).mean())
    df = df.merge(pd.DataFrame({"Date": tr["Date"], "symbol": tr["symbol"], "avg5": avg5}), on=["Date", "symbol"], how="left")
    df["avg5"] = df.groupby("symbol")["avg5"].ffill()

    # forward fill only: the old bfill used FUTURE macro values, which inference never has
    df["idx_macro_close"] = df.groupby("symbol")["idx_macro_close"].ffill()

    n0 = len(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)      # newest session(s) / suspensions: nothing to learn from
    print(f"   {n0 - len(df)} rows without a target dropped (newest session(s) and suspensions); {len(df)} rows, "
          f"{df['symbol'].nunique()} companies, {df['Date'].min():%Y-%m-%d} .. {df['Date'].max():%Y-%m-%d}")
    if symbols:
        want = {s.strip().upper().replace(".JK", "") for s in symbols}
        df = df[df["symbol"].str.replace(".JK", "", regex=False).str.upper().isin(want)].reset_index(drop=True)
        if df.empty:
            sys.exit(f"[!] none of {sorted(want)} found in {csv_path}")
    return df


def time_blocks(df):
    """One set of date cut-offs for ALL companies (no company's future leaks into another's training)."""
    dates = np.sort(df["Date"].unique())
    d1 = pd.Timestamp(dates[max(int(len(dates) * TRAIN_FRAC) - 1, 0)])
    d2 = pd.Timestamp(dates[max(int(len(dates) * (TRAIN_FRAC + VAL_FRAC)) - 1, 0)])
    emb = pd.Timedelta(days=EMBARGO_DAYS)
    blocks = {"train": (df["Date"] <= d1).to_numpy(),
              "val": ((df["Date"] > d1 + emb) & (df["Date"] <= d2)).to_numpy(),
              "test": (df["Date"] > d2 + emb).to_numpy()}
    return blocks, {"train_end": f"{d1:%Y-%m-%d}", "val_start": f"{d1 + emb:%Y-%m-%d}", "val_end": f"{d2:%Y-%m-%d}",
                    "test_start": f"{d2 + emb:%Y-%m-%d}", "embargo_days": EMBARGO_DAYS}


def dmat(d, with_label=True):
    X = d[FEATURES]
    return xgb.DMatrix(X, label=np.log1p(d[TARGET].to_numpy(float)) if with_label else None, missing=np.nan)


def predict_volume(booster, d):
    return np.clip(np.expm1(booster.predict(dmat(d, with_label=False))), 0, None)


# ---- boosting helpers ---------------------------------------------------------------------------------------------
def pick_rounds(params, dtrain, dval, max_rounds, patience, base_path=None):
    """Number of boosting rounds with the lowest validation error (early stopping decides when to give up)."""
    history = {}
    xgb.train(params, dtrain, num_boost_round=max_rounds, evals=[(dval, "val")], evals_result=history,
              early_stopping_rounds=patience, verbose_eval=False, xgb_model=base_path)
    curve = np.asarray(history["val"][params["eval_metric"]], float)
    return int(np.argmin(curve)) + 1


def fit_fixed(params, dtrain, rounds, base_path=None):
    return xgb.train(params, dtrain, num_boost_round=int(rounds), verbose_eval=False, xgb_model=base_path)


# ---- main ---------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the XGBoost next-session volume models (see the module docstring).")
    paths = config.Paths()
    ap.add_argument("--csv", default=paths.output_path(paths.mcs_features_csv), help="MCS_features.csv (output of `pipeline.py prepare`)")
    ap.add_argument("--model-dir", default=paths.xgboost_models_folder, help="where the models are written")
    ap.add_argument("--symbols", default=None, help="comma separated companies to fine-tune, e.g. GIAA,SMDR (the base model still uses all)")
    ap.add_argument("--no-refit", action="store_true", help="deploy the models trained on train+validation only (default: refit on all rows)")
    ap.add_argument("--smoke", action="store_true", help="quick sanity run: 2 companies, few rounds, no refit")
    a = ap.parse_args(argv)

    global ROUNDS_MAX, TUNE_ROUNDS_MAX
    t0 = time.time()
    print(f"=== XGBoost trainer | target = {config.XGB_TARGET_MODE} | csv = {a.csv}")
    df = load_frame(a.csv)
    syms = sorted(df["symbol"].unique())
    if a.symbols:
        want = {s.strip().upper().replace(".JK", "") for s in a.symbols.split(",")}
        syms = [s for s in syms if s.replace(".JK", "").upper() in want]
    if a.smoke:
        syms = syms[:2]
        ROUNDS_MAX, TUNE_ROUNDS_MAX = 40, 10
        a.no_refit = True
        print("   --smoke: 2 companies, few rounds, no refit (numbers are NOT meaningful)")

    blocks, cuts = time_blocks(df)
    print(f"   split: train <= {cuts['train_end']} | validation {cuts['val_start']}..{cuts['val_end']} | test >= {cuts['test_start']}"
          f"  (embargo {EMBARGO_DAYS} d)  rows: " + ", ".join(f"{k} {int(m.sum())}" for k, m in blocks.items()))
    tr, va, te = (df[blocks[k]] for k in ("train", "val", "test"))
    if len(tr) < 500 or len(va) < 100:
        sys.exit("[!] too little history for a train/validation/test split.")

    # ---- 1) pooled base model: rounds by validation, then a model trained on the train block only ------------------
    print("\n[1/3] Base model (pooled over all companies)")
    dtr, dva = dmat(tr), dmat(va)
    rounds_base = pick_rounds(BASE_PARAMS, dtr, dva, ROUNDS_MAX, PATIENCE)
    base_A = fit_fixed(BASE_PARAMS, dtr, rounds_base)
    print(f"   {rounds_base} rounds (early stopping on the validation block)")

    tmp = tempfile.mkdtemp(prefix="xgb_train_")
    base_A_path = os.path.join(tmp, "base_A.json")
    base_A.save_model(base_A_path)

    def evaluate(booster_or_pred, d):
        d = d[d["Volume"] > 0]                                    # the rows the dashboard shows: today traded
        p = predict_volume(booster_or_pred, d) if hasattr(booster_or_pred, "predict") else booster_or_pred
        m = level_metrics(d[TARGET], p)
        m["rise_fall_corr"], m["naive_rise_fall_corr"] = rise_fall_corr(d, p)
        m["naive_today_WMAPE_%"] = wmape(d[TARGET], d["Volume"])
        dd = d.dropna(subset=["avg5"])
        m["naive_avg5_WMAPE_%"] = wmape(dd[TARGET], dd["avg5"]) if len(dd) else np.nan
        m["naive_avg5_SMAPE_%"] = smape(dd[TARGET], dd["avg5"]) if len(dd) else np.nan
        return m

    rep_val, rep_test = evaluate(base_A, va), evaluate(base_A, te)
    print("   pooled hold-out (rows where the stock traded that day):")
    tab = pd.DataFrame({"validation": rep_val, "TEST": rep_test}).T
    print(tab[["n", "WMAPE_%", "SMAPE_%", "bias", "rise_fall_corr", "naive_rise_fall_corr", "naive_today_WMAPE_%", "naive_avg5_WMAPE_%", "naive_avg5_SMAPE_%"]]
          .round(2).to_string())
    if not rep_test["WMAPE_%"] < rep_test["naive_avg5_WMAPE_%"]:
        print("   ⚠️ On the TEST block the base model is NOT more accurate than 'mean of the last 5 sessions'. "
              "Treat the forecast LEVEL with caution (the rise/fall signal may still carry information).")

    # ---- 2) per-company fine-tune, kept only if it wins on validation ---------------------------------------------
    print("\n[2/3] Per-company fine-tune (kept only if it beats the base model on validation)")
    decisions, tune_rounds, rows = {}, {}, []
    for sym in syms:
        c = {k: d[d["symbol"] == sym] for k, d in (("train", tr), ("val", va), ("test", te))}
        base_val, base_test = evaluate(base_A, c["val"]) if len(c["val"]) else None, evaluate(base_A, c["test"]) if len(c["test"]) else None
        row = {"symbol": sym, "rows_train": len(c["train"]), "rows_val": len(c["val"]), "rows_test": len(c["test"])}
        if len(c["train"]) < MIN_TRAIN_ROWS or len(c["val"]) < MIN_VAL_ROWS:
            decisions[sym] = "base (too little history for a fine-tune)"
        else:
            dtr_c, dva_c = dmat(c["train"]), dmat(c["val"])
            r_c = pick_rounds(TUNE_PARAMS, dtr_c, dva_c, TUNE_ROUNDS_MAX, TUNE_PATIENCE, base_path=base_A_path)
            tuned_A = fit_fixed(TUNE_PARAMS, dtr_c, r_c, base_path=base_A_path)
            tun_val, tun_test = evaluate(tuned_A, c["val"]), evaluate(tuned_A, c["test"])
            wins = tun_val["WMAPE_%"] < base_val["WMAPE_%"] * (1.0 - MIN_IMPROVEMENT)
            decisions[sym] = "fine-tuned" if wins else "base"
            tune_rounds[sym] = r_c
            row.update({"tuned_val_WMAPE_%": tun_val["WMAPE_%"], "tuned_test_WMAPE_%": tun_test["WMAPE_%"],
                        "tuned_test_rise_fall_corr": tun_test["rise_fall_corr"]})
        row.update({"base_val_WMAPE_%": base_val["WMAPE_%"] if base_val else np.nan,
                    "base_test_WMAPE_%": base_test["WMAPE_%"] if base_test else np.nan,
                    "base_test_bias": base_test["bias"] if base_test else np.nan,
                    "base_test_rise_fall_corr": base_test["rise_fall_corr"] if base_test else np.nan,
                    "naive_avg5_test_WMAPE_%": base_test["naive_avg5_WMAPE_%"] if base_test else np.nan,
                    "deployed": decisions[sym]})
        rows.append(row)
        print(f"   {sym:<9} val WMAPE base {row['base_val_WMAPE_%']:6.1f} -> tuned {row.get('tuned_val_WMAPE_%', float('nan')):6.1f}   => {decisions[sym]}")
    report = pd.DataFrame(rows)

    # ---- 3) final models: refit on all rows with the rounds chosen above ------------------------------------------
    print("\n[3/3] Final models")
    final_mask = np.ones(len(df), bool) if not a.no_refit else (blocks["train"] | blocks["val"])
    d_final = df[final_mask]
    os.makedirs(a.model_dir, exist_ok=True)
    base_F = fit_fixed(BASE_PARAMS, dmat(d_final), rounds_base)
    base_path = os.path.join(a.model_dir, paths.xgboost_base_model_filename)
    base_F.save_model(base_path)
    print(f"   base model -> {base_path}  ({'all rows' if not a.no_refit else 'train+validation rows'}, {rounds_base} rounds)")
    for sym in syms:
        out = os.path.join(a.model_dir, paths.xgboost_model_filename_tpl.format(company=sym.replace('.JK', '')))
        if decisions[sym] == "fine-tuned":
            d_c = d_final[d_final["symbol"] == sym]
            fit_fixed(TUNE_PARAMS, dmat(d_c), tune_rounds[sym], base_path=base_path).save_model(out)
            print(f"   fine-tuned -> {out}  ({tune_rounds[sym]} extra rounds)")
        elif os.path.exists(out):
            os.remove(out)                            # a stale (old-target) model would otherwise shadow the base model
            print(f"   removed stale {out} (the base model is used for {sym})")

    meta = {"version": 2, "target_mode": config.XGB_TARGET_MODE, "target_max_gap_days": config.TARGET_MAX_GAP_DAYS,
            "features": FEATURES, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "split": cuts,
            "deployed_on": "all rows" if not a.no_refit else "train+validation rows",
            "rounds": {"base": rounds_base, "fine_tune": tune_rounds}, "decisions": decisions,
            "holdout_pooled": {"validation": rep_val, "test": rep_test,
                               "note": "measured on the models BEFORE the final refit; rows = days the stock traded"}}
    with open(os.path.join(a.model_dir, config.XGB_META_FILENAME), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=float)
    report.round(3).to_csv(os.path.join(a.model_dir, "xgb_holdout_report.csv"), index=False)
    shutil.rmtree(tmp, ignore_errors=True)

    print("\n=== HOLD-OUT REPORT (TEST block; WMAPE in %, lower is better) ===")
    show = report[["symbol", "rows_test", "base_test_WMAPE_%", "tuned_test_WMAPE_%", "naive_avg5_test_WMAPE_%", "base_test_bias",
                   "base_test_rise_fall_corr", "deployed"]]
    print(show.round(2).to_string(index=False))
    print(f"\nDone in {time.time() - t0:.0f}s. Next: `python pipeline.py score`. "
          f"{'(--smoke numbers are not meaningful.)' if a.smoke else ''}")


if __name__ == "__main__":
    main()
