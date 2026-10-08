#!/usr/bin/env python3
"""
lstm_hurst_train.py
--------------------
Trains the breakout-probability model: P(Close exceeds its 252-session high within the next 21 sessions), from
scale-free price features that include the Hurst exponent (see lstm_features.py for the exact definitions).

Run it from the pipeline/ folder, after `python pipeline.py prepare` (it reads MCS_features.csv):

    python lstm_hurst_train.py                     # ONE pooled model for every company  (default)
    python lstm_hurst_train.py --mode both         # pooled + a model per company (kept only where it beats the pooled one)
    python lstm_hurst_train.py --no-lstm           # logistic model only (no torch needed)
    python lstm_hurst_train.py --smoke             # ~1 minute sanity check, numbers are NOT meaningful
    python lstm_hurst_train.py --deploy logit|lstm # force the model type instead of letting validation decide
    python lstm_hurst_train.py --refit-all         # refit on ALL rows after evaluating (see below)

Needs `pip install hurst` (the Hurst feature) and, unless --no-lstm, torch. Run `pip install hurst torch scikit-learn`.

Output, per group, in <lstm_hurst_model_path>/<_POOLED or COMPANY>/ (lstm_hurst_predictor.py reads exactly this):
    lstm_hurst_best_params.json     features, scaler, logistic coefficients, calibration, metrics, which model won
    best_lstm_hurst_model.pth       only when the LSTM won
    breakout_holdout_report.csv     per-company hold-out numbers of the deployed model

WHAT CHANGED versus the old trainer, and why (measured on the 10-company dataset)
  * HELD-OUT, not memorised. The old per-company LSTMs had train AUC 0.998 but 0.877 on their hold-out, were worse
    than "always predict the base rate" on 6 of 8 companies (BIRD said 46% while 11% happened, SMDR 0.1% while 15.5%
    happened) and were beaten by the one-line rule "closer to the high -> more likely" (AUC 0.95).
  * SCALE-FREE features instead of the raw Close level (up to 24 sigma out of range for ELPI) -- which is also what
    makes a POOLED model possible: with ~10x the data the logistic model reaches test AUC 0.92 / Brier 0.08, the old
    deployed LSTMs scored AUC 0.75 / Brier 0.19 on the same rows (worse than the 0.147 base-rate forecast).
  * TRADING-DAY windows (252-session high, 21-session horizon, 100-session Hurst) instead of calendar rows.
  * Splits: train 60% | validation 20% | test 20%, by DATE (one set of cut-offs for every company, so no company's
    future leaks into another's training), PURGED so that no label looks into a later block (21 sessions span 29-48
    calendar days, so a fixed gap is not enough) plus a 7-day embargo. Validation chooses the epochs, hyper-parameters,
    the model type and fits the calibration; TEST is only reported. (The old script picked the grid winner by its TEST AUC,
    ran a fixed 30 epochs, and had no embargo.)
  * CALIBRATION: Platt scaling fitted on the validation block (the base rate moved 8% -> 17% between periods).
  * HEAD-TO-HEAD: the LSTM is deployed only if its validation Brier beats the logistic model's by >= 2%; otherwise
    the logistic model is. The result is compared with a constant-base-rate forecast and the one-feature rule on TEST;
    `beats_baseline` in the params file says whether the deployed model won, and lstm_hurst_predictor.py skips a model
    that did not (config.LSTM_USE_ONLY_IF_BEATS_BASELINE).
  * --refit-all: refit the chosen model on every row after the evaluation, WITHOUT calibration (a calibration fitted on
    the train-only model would not transfer). The reported metrics are those of the train-only model.
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    from . import config
    from . import lstm_features as lf
except ImportError:  # running as a plain script, not as a package
    import config
    import lstm_features as lf

# ---- settings (edit here) ---------------------------------------------------------------------------------------
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2
MIN_SAMPLES = {"train": 300, "val": 60, "test": 60}       # a group with fewer samples in a block is skipped
MIN_TRAIN_POSITIVES = 20                                  # ...and so is one with fewer than this many breakouts (or non-breakouts) to learn from
GRID = [{"hidden_size": h, "lr": lr} for h in (16, 32) for lr in (1e-3, 3e-3)]
MAX_EPOCHS, PATIENCE, BATCH_SIZE, DROPOUT = 120, 12, 64, 0.3
LSTM_MUST_BEAT_LOGIT_BY = 0.02                              # validation Brier, relative
GATE_SLACK = 1.02                                           # deployed model may be at most 2% worse than the one-feature rule
EMBARGO_DAYS = 7                                            # calendar days dropped after each boundary (see lf.split_cutoffs)
OWN_MUST_BEAT_POOLED_BY = 0.02                              # a company's own model must cut the pooled model's test Brier by >= 2%


def log(msg=""):
    print(msg, flush=True)


def assemble(symbols, samples):
    """Concatenate the samples of several companies (time order inside each company is kept)."""
    parts = [(s, samples[s]) for s in symbols if len(samples[s]["y"])]
    if not parts:
        return None
    return {"X": np.concatenate([p["X"] for _, p in parts]), "y": np.concatenate([p["y"] for _, p in parts]),
            "date": np.concatenate([p["date"] for _, p in parts]),
            "label_end": np.concatenate([p["label_end"] for _, p in parts]),
            "symbol": np.concatenate([np.full(len(p["y"]), s) for s, p in parts])}


def remove_stale(out_dir):
    """Delete the artifacts of an EARLIER run of THIS trainer (feature_version 2) so they cannot shadow the pooled model."""
    if lf.read_params(out_dir).get("feature_version") != config.LSTM_FEATURE_VERSION:
        return                                              # not ours (e.g. an old-method folder): leave it alone
    for f in (config.LSTM_PARAMS_FILENAME, config.LSTM_MODEL_FILENAME, "breakout_holdout_report.csv"):
        if os.path.exists(os.path.join(out_dir, f)):
            os.remove(os.path.join(out_dir, f))
    if not os.listdir(out_dir):
        os.rmdir(out_dir)
    log(f"   removed the stale model in {out_dir}")


def train_group(name, symbols, samples, cuts, args, lm, pooled_brier=None):
    """
    Train / validate / test one group (the pooled group or a single company).
    Returns (params, per-company hold-out report) or (None, None) when the group is skipped or not worth keeping.
    `pooled_brier` {symbol: test Brier of the pooled model}: a company's OWN model is kept only if it beats that.
    """
    out_dir = os.path.join(args.model_dir, name)
    data = assemble(symbols, samples)
    if data is None:
        log(f"[{name}] no samples -> skipped")
        remove_stale(out_dir)
        return None, None
    feats = config.LSTM_FEATURES
    masks = lf.block_masks(data["date"], cuts, data["label_end"])
    n = {k: int(m.sum()) for k, m in masks.items()}
    if any(n[k] < MIN_SAMPLES[k] for k in n):
        log(f"[{name}] too few samples {n} (need {MIN_SAMPLES}) -> skipped")
        remove_stale(out_dir)
        return None, None
    tr, va, te = ({k: data[k][m] for k in ("X", "y", "date", "symbol")} for m in (masks["train"], masks["val"], masks["test"]))
    log(f"\n=== [{name}] {len(symbols)} companies | samples train {n['train']} / validation {n['val']} / test {n['test']} | "
        f"breakout rate train {tr['y'].mean():.1%} / val {va['y'].mean():.1%} / test {te['y'].mean():.1%}")
    n_pos = int(tr["y"].sum())
    if n_pos < MIN_TRAIN_POSITIVES or n_pos > len(tr["y"]) - MIN_TRAIN_POSITIVES:
        log(f"[{name}] only {n_pos} breakouts in {len(tr['y'])} training samples (need >= {MIN_TRAIN_POSITIVES} of each outcome) -> skipped")
        remove_stale(out_dir)
        return None, None

    scaler = lf.fit_scaler(tr["X"].reshape(-1, tr["X"].shape[2]))
    Z = {k: lf.standardize(d["X"][:, -1, :], scaler) for k, d in (("train", tr), ("val", va), ("test", te))}   # last session only
    i_sig = feats.index("dist_res_sigma")

    # --- baselines and the logistic candidate -----------------------------------------------------------------
    logit_all = lf.fit_logit(Z["train"], tr["y"])
    logit_sig = lf.fit_logit(Z["train"][:, [i_sig]], tr["y"])
    z_all = {k: lf.logit_scores(logit_all, Z[k]) for k in Z}
    z_sig = {k: lf.logit_scores(logit_sig, Z[k][:, [i_sig]]) for k in Z}
    cal_all, cal_sig = lf.fit_platt(z_all["val"], va["y"]), lf.fit_platt(z_sig["val"], va["y"])
    cand = {"logit": {"z": z_all, "cal": cal_all}, "logit_sigma_only": {"z": z_sig, "cal": cal_sig}}
    const = {"base_rate_train": float(tr["y"].mean()), "base_rate_val": float(va["y"].mean())}

    # --- the LSTM candidate -----------------------------------------------------------------------------------
    lstm_best = None
    if lm is not None:
        Xs = {k: lf.standardize(d["X"], scaler).astype(np.float32) for k, d in (("train", tr), ("val", va), ("test", te))}
        grid = GRID[:1] if args.smoke else GRID
        max_epochs = 3 if args.smoke else MAX_EPOCHS
        for cfg in grid:
            model, epoch, vloss = lm.fit_lstm(Xs["train"], tr["y"], Xs["val"], va["y"], cfg["hidden_size"], cfg["lr"],
                                              max_epochs=max_epochs, patience=PATIENCE, batch_size=BATCH_SIZE, dropout=DROPOUT,
                                              seed=config.RANDOM_SEED)
            log(f"   LSTM hidden={cfg['hidden_size']:>2} lr={cfg['lr']:g}: best epoch {epoch:>3}, validation loss {vloss:.4f}")
            if lstm_best is None or vloss < lstm_best["val_loss"]:
                lstm_best = {"model": model, "cfg": cfg, "epoch": epoch, "val_loss": vloss}
        z_lstm = {k: lm.predict_logits(lstm_best["model"], Xs[k]) for k in Xs}
        cand["lstm"] = {"z": z_lstm, "cal": lf.fit_platt(z_lstm["val"], va["y"])}

    # --- metrics: validation and test, every candidate calibrated the same way ---------------------------------------
    metrics = {"val": {}, "test": {}}
    for nm, c in cand.items():
        for k, d in (("val", va), ("test", te)):
            metrics[k][nm] = lf.binary_metrics(d["y"], lf.apply_platt(c["z"][k], c["cal"]))
    for k, d in (("val", va), ("test", te)):
        for nm, rate in const.items():
            metrics[k][nm] = lf.binary_metrics(d["y"], np.full(len(d["y"]), rate))

    # --- choose what to deploy (validation only) ------------------------------------------------------------------
    if args.deploy in ("logit", "lstm"):
        chosen = args.deploy
        if chosen == "lstm" and "lstm" not in cand:
            sys.exit("[!] --deploy lstm needs torch (or drop --no-lstm)")
    else:
        chosen = "logit"
        if "lstm" in cand and metrics["val"]["lstm"]["brier"] < metrics["val"]["logit"]["brier"] * (1 - LSTM_MUST_BEAT_LOGIT_BY):
            chosen = "lstm"
    final_test = metrics["test"][chosen]
    beats = bool(final_test["brier"] <= metrics["test"]["base_rate_val"]["brier"] and
                 final_test["brier"] <= metrics["test"]["logit_sigma_only"]["brier"] * GATE_SLACK)

    log("   validation / test (calibrated; lower Brier is better):")
    rows = []
    for nm in list(cand) + list(const):
        rows.append({"model": nm, "val_AUC": metrics["val"][nm]["auc"], "val_Brier": metrics["val"][nm]["brier"],
                     "test_AUC": metrics["test"][nm]["auc"], "test_Brier": metrics["test"][nm]["brier"],
                     "test_logloss": metrics["test"][nm]["logloss"], "test_ECE": metrics["test"][nm]["ece"]})
    log(pd.DataFrame(rows).set_index("model").round(3).to_string())
    log(f"   -> deployed model: {chosen.upper()}   |   beats the baselines on TEST: {'YES' if beats else 'NO'}"
        + ("" if beats else "   (the predictor will skip it unless config.LSTM_USE_ONLY_IF_BEATS_BASELINE is False)"))
    if len(symbols) == 1 and name != config.LSTM_POOLED_DIRNAME and not beats:
        log("   -> its own model does not beat the baselines on this company's test block -> not kept; the pooled model is used")
        remove_stale(out_dir)
        return None, None
    if pooled_brier is not None and len(symbols) == 1 and name != config.LSTM_POOLED_DIRNAME:
        ref = pooled_brier.get(symbols[0])
        if ref is not None and not final_test["brier"] < ref * (1.0 - OWN_MUST_BEAT_POOLED_BY):
            log(f"   -> its own model (test Brier {final_test['brier']:.3f}) does not beat the pooled model on this company "
                f"({ref:.3f}) -> not kept; the pooled model is used")
            remove_stale(out_dir)
            return None, None

    # --- final artifacts ----------------------------------------------------------------------------------------------
    deployed_on = "train" if not args.refit_all else "all"
    scaler_out, logit_out, platt_logit, platt_lstm = scaler, dict(logit_all), cal_all, cand.get("lstm", {}).get("cal")
    lstm_out = None
    if args.refit_all:                                       # no calibration: it would not transfer to a refit model
        X_all = np.concatenate([tr["X"], va["X"], te["X"]]); y_all = np.concatenate([tr["y"], va["y"], te["y"]])
        scaler_out = lf.fit_scaler(X_all.reshape(-1, X_all.shape[2]))
        logit_out = lf.fit_logit(lf.standardize(X_all[:, -1, :], scaler_out), y_all)
        platt_logit, platt_lstm = {"a": 1.0, "b": 0.0, "fitted": False}, {"a": 1.0, "b": 0.0, "fitted": False}
        if chosen == "lstm":
            lstm_best["model"] = lm.fit_lstm_fixed(lf.standardize(X_all, scaler_out).astype(np.float32), y_all, lstm_best["cfg"]["hidden_size"],
                                                   lstm_best["cfg"]["lr"], lstm_best["epoch"], batch_size=BATCH_SIZE, dropout=DROPOUT,
                                                   seed=config.RANDOM_SEED)
    if chosen == "lstm":
        lstm_out = {"hidden_size": lstm_best["cfg"]["hidden_size"], "lr": lstm_best["cfg"]["lr"], "best_epoch": lstm_best["epoch"],
                    "dropout": DROPOUT, "calibration": platt_lstm}

    params = {"feature_version": config.LSTM_FEATURE_VERSION, "model_type": chosen, "group": name, "symbols": list(symbols),
              "features": feats, "seq_length": config.LSTM_SEQ_LENGTH, "horizon": config.LSTM_HORIZON,
              "yearly_window": config.LSTM_YEARLY_WINDOW, "yearly_min_periods": config.LSTM_YEARLY_MIN_PERIODS,
              "vol_window": config.LSTM_VOL_WINDOW, "hurst_window": config.HURST_WINDOW, "max_abs_log_ret": config.LSTM_MAX_ABS_LOG_RET,
              "scaler": scaler_out, "logit": logit_out, "calibration": platt_logit, "lstm": lstm_out,
              "split": {"train_end": f"{cuts['d1']:%Y-%m-%d}", "val_end": f"{cuts['d2']:%Y-%m-%d}", "embargo_days": cuts["embargo"].days,
                        "purged_by_label_end": True},
              "deployed_on": deployed_on, "beats_baseline": beats, "metrics": metrics,
              "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    lf.write_params(out_dir, params)
    if chosen == "lstm":
        lm.save_lstm(lstm_best["model"], os.path.join(out_dir, config.LSTM_MODEL_FILENAME))
    elif os.path.exists(os.path.join(out_dir, config.LSTM_MODEL_FILENAME)):
        os.remove(os.path.join(out_dir, config.LSTM_MODEL_FILENAME))      # a stale .pth must not shadow the logistic model
    log(f"   saved -> {out_dir}  (trained on {deployed_on} rows)")

    # per-company breakdown of the deployed model on TEST
    if chosen == "lstm":
        p_te = lf.apply_platt(cand["lstm"]["z"]["test"], cand["lstm"]["cal"])
    else:
        p_te = lf.apply_platt(cand["logit"]["z"]["test"], cand["logit"]["cal"])
    rep = []
    for sym in sorted(set(te["symbol"])):
        m = te["symbol"] == sym
        mm = lf.binary_metrics(te["y"][m], p_te[m])
        rep.append({"symbol": sym, "n_test": mm["n"], "breakout_rate": mm["base_rate"], "mean_P": mm["mean_p"], "AUC": mm["auc"], "Brier": mm["brier"],
                    "Brier_constant": float(np.mean((te["y"][m] - const["base_rate_val"]) ** 2))})
    rep = pd.DataFrame(rep)
    rep.round(3).to_csv(os.path.join(out_dir, "breakout_holdout_report.csv"), index=False)
    if name == config.LSTM_POOLED_DIRNAME:
        log("   deployed model on the TEST block, per company:")
        log(rep.round(3).to_string(index=False))
    return params, rep


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the breakout-probability model (see the module docstring).")
    paths = config.Paths()
    ap.add_argument("--csv", default=paths.output_path(paths.mcs_features_csv), help="MCS_features.csv (output of `pipeline.py prepare`)")
    ap.add_argument("--model-dir", default=paths.lstm_hurst_model_path, help="folder that receives _POOLED/ and/or <COMPANY>/")
    ap.add_argument("--mode", choices=["pooled", "per-company", "both"], default="pooled")
    ap.add_argument("--no-lstm", action="store_true", help="logistic model only (no torch needed)")
    ap.add_argument("--deploy", choices=["auto", "logit", "lstm"], default="auto")
    ap.add_argument("--refit-all", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    t0 = time.time()
    lm = None
    if not a.no_lstm:
        try:
            try:
                from . import lstm_model as lm
            except ImportError:
                import lstm_model as lm
        except ImportError as e:
            log(f"[i] torch is not available ({e}) -> training the logistic model only (--no-lstm)")
            lm = None
    if lf.get_hurst_estimator() is None:
        sys.exit("[!] The `hurst` package is required for the Hurst feature: pip install hurst")

    log(f"=== breakout model trainer | csv = {a.csv}")
    df = pd.read_csv(a.csv, parse_dates=["Date"])
    for c in ("Date", "symbol", "Close", "Volume"):
        if c not in df.columns:
            sys.exit(f"[!] {a.csv} lacks column {c!r}")
    log("[1/3] Features and labels (trading-day windows; this computes the Hurst exponent, ~1 minute)")
    frames = lf.build_symbol_frames(df, with_label=True, verbose=True)
    samples = {s: lf.build_samples(fr) for s, fr in frames.items()}
    for s, smp in samples.items():
        log(f"   {s:<9} {len(frames[s]):>5} sessions -> {len(smp['y']):>5} labelled sequences")
    all_dates = np.concatenate([s["date"] for s in samples.values() if len(s["y"])])
    cuts = lf.split_cutoffs(all_dates, TRAIN_FRAC, VAL_FRAC, EMBARGO_DAYS)
    log(f"[2/3] Split by date: train <= {cuts['d1']:%Y-%m-%d} | validation up to {cuts['d2']:%Y-%m-%d} | test after "
        f"(PURGED: no label looks past its block; + {EMBARGO_DAYS}-day embargo)")

    groups = {}
    if a.mode in ("pooled", "both"):
        groups[config.LSTM_POOLED_DIRNAME] = list(samples)
    if a.mode in ("per-company", "both"):
        for s in samples:
            groups[s.replace(".JK", "")] = [s]
    if a.smoke:
        groups = dict(list(groups.items())[:1])
        log("   --smoke: first group only, 1 LSTM config, 3 epochs (numbers are NOT meaningful)")
    log("[3/3] Training")
    results, pooled_brier = {}, None
    for name, syms in groups.items():                       # the pooled group (if any) runs first: it is the yardstick for the rest
        params, rep = train_group(name, syms, samples, cuts, a, lm, pooled_brier)
        results[name] = params
        if name == config.LSTM_POOLED_DIRNAME and rep is not None:
            pooled_brier = dict(zip(rep["symbol"], rep["Brier"]))
    trained = {k: v for k, v in results.items() if v}
    summary = ", ".join(
    f"{k} ({v['model_type']}, beats baseline: {v['beats_baseline']})"
    for k, v in trained.items()) or "nothing"
    log(f"\nDone in {time.time() - t0:.0f}s. Trained: {summary}. "
    "Next: `python pipeline.py score`.")
    return results


if __name__ == "__main__":
    main()
