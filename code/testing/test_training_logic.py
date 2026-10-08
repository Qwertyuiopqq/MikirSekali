"""
test_training_logic.py -- executable checks of the training-side rules (target, features, labels, splits, calibration).

Run from the project folder (the one containing pipeline/, website/ and testing/):

    python testing/test_training_logic.py
    python -m pytest testing/test_training_logic.py -v

Needs numpy / pandas / scikit-learn only. The tests of train_xgboost_transfer.py are skipped when `xgboost` is missing;
nothing here needs torch or the `hurst` package (a cheap deterministic stand-in estimator is injected).
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import config                                   # noqa: E402
from pipeline import lstm_features as lf                       # noqa: E402
from pipeline.fuzzy_system import xgb_applicable               # noqa: E402
from pipeline.spine_builder import next_session_target, _fill_and_flag   # noqa: E402


class Skip(Exception):
    pass


def skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        raise Skip(msg)


def stub_hurst(window):
    """Cheap deterministic stand-in for the `hurst` package (lag-1 autocorrelation of the returns -> 0.2..0.8)."""
    r = np.diff(window) / window[:-1]
    c = np.corrcoef(r[:-1], r[1:])[0, 1] if len(r) > 3 else 0.0
    return float(0.5 + 0.3 * np.tanh(5 * (0.0 if np.isnan(c) else c)))


lf.set_hurst_estimator(stub_hurst)


def make_spine(symbols=("AAA.JK", "BBB.JK", "CCC.JK"), start="2021-01-04", days=1400, seed=0):
    """Calendar spine like the pipeline's: every day, weekends/holidays with Volume 0 and forward-filled prices."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    frames = []
    for k, sym in enumerate(symbols):
        price = (100 + 40 * k) * np.exp(np.cumsum(rng.normal(0.0006, 0.02, days)))
        vol = rng.lognormal(15, 0.6, days)
        closed = (dates.dayofweek >= 5) | (rng.random(days) < 0.03)
        vol[closed] = 0.0
        close = pd.Series(np.where(closed, np.nan, price)).ffill().bfill()
        frames.append(pd.DataFrame({"Date": dates, "symbol": sym, "Close": close.to_numpy(), "Volume": vol, "day_of_week": dates.dayofweek}))
    return pd.concat(frames, ignore_index=True)


# ======================================================================================
# The forecast target and the spine
# ======================================================================================
def test_target_is_the_volume_of_the_next_trading_session():
    dates = pd.date_range("2026-09-14", "2026-09-28", freq="D")           # Mon .. Mon
    vol_a = [10, 20, 30, 40, 50, 0, 0, 60, 70, 0, 90, 100, 0, 0, 110]      # Wed 09-23 is a holiday
    vol_b = [10, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5]                # suspended 09-16 .. 09-27
    df = pd.concat([pd.DataFrame({"Date": dates, "symbol": s, "Volume": v}) for s, v in (("AAA.JK", vol_a), ("BBB.JK", vol_b))])
    df = df.sort_values(["symbol", "Date"]).reset_index(drop=True)
    t = next_session_target(df)
    a = t[df.symbol == "AAA.JK"].to_numpy()
    nan = np.nan
    assert np.allclose(a, [20, 30, 40, 50, 60, 60, 60, 70, 90, 90, 100, 110, 110, 110, nan], equal_nan=True)
    # Friday -> MONDAY (not the weekend's 0), holiday -> the day after, the newest session has no target yet
    b = t[df.symbol == "BBB.JK"].to_numpy()
    assert np.isnan(b[1])                         # next session is 13 days away: a suspension, not a "next day" forecast
    assert b[8] == 5.0                            # within 7 days of the day trading resumes
    assert not (t == 0).any()


def test_a_zero_volume_corporate_action_record_does_not_carry_its_own_price():
    # TMAS 2023-05-23: a split record (Volume 0) priced 38.5 beside a real ~265 made the next session look like +560%
    d = pd.date_range("2023-05-19", "2023-05-26")
    raw = pd.DataFrame({"Date": d, "symbol": "TMAS.JK", "Open": [253, np.nan, np.nan, 265, 38.5, 255, 264, 264],
                        "High": [253, np.nan, np.nan, 265, 38.5, 255, 264, 264], "Low": [253, np.nan, np.nan, 265, 38.5, 255, 264, 264],
                        "Close": [253, np.nan, np.nan, 265, 38.5, 255, 264, 264],
                        "Volume": [2.6e6, np.nan, np.nan, 1e7, 0.0, 9.1e6, 6.5e6, np.nan],
                        "Dividends": np.nan, "Stock Splits": [np.nan] * 4 + [10.0] + [np.nan] * 3})
    out = _fill_and_flag(raw.copy())
    assert out["Close"].iloc[4] == 265.0 and out["Stock Splits"].iloc[4] == 10.0       # price carried forward, the split is still recorded
    assert abs(out["Close"].iloc[5] / out["Close"].iloc[4] - 1) < 0.1


def test_forecast_usable_rows_follow_the_target_definition():
    d = pd.DataFrame({"day_of_week": [4, 5, 6], "Volume": [1, 0, 0]})
    assert xgb_applicable(d).tolist() == [False, False, False]                         # old definition: weekend forecasts are ~0
    assert xgb_applicable(d.assign(xgb_target_mode="next_trading_day")).tolist() == [True, True, True]


# ======================================================================================
# Breakout-model features and labels
# ======================================================================================
def test_chain_link_removes_a_split_jump_and_leaves_clean_series_alone():
    clean = np.array([100.0, 102, 101, 103, 104])
    c, n = lf.chain_link(clean)
    assert n == 0 and np.allclose(c, clean)
    jumpy = np.array([100.0, 102, 101, 10.3, 10.4, 10.2])                              # a 1:10 consolidation after the 3rd price
    c, n = lf.chain_link(jumpy)
    k = 10.3 / 101.0
    assert n == 1
    assert np.allclose(c[3:], [10.3, 10.4, 10.2])                                       # prices after the jump are untouched
    assert np.allclose(c[:3], np.array([100.0, 102, 101]) * k)                          # earlier ones rescaled: the jump is gone
    assert abs(c[3] / c[2] - 1.0) < 1e-12 and abs(c[1] / c[0] - 1.02) < 1e-12           # ...and their own returns are preserved


def test_breakout_label_matches_a_naive_loop():
    g = make_spine(("AAA.JK",), days=900, seed=3)
    fr = lf.symbol_frame(g, with_label=True)
    close, H, W = fr["close_adj"].to_numpy(), config.LSTM_HORIZON, config.LSTM_YEARLY_WINDOW
    for t in (80, 150, 400, 600, len(fr) - H - 1, len(fr) - H, len(fr) - 1):
        lo = max(0, t - W + 1)
        res = close[lo:t + 1].max() if t + 1 >= config.LSTM_YEARLY_MIN_PERIODS else np.nan
        fut = close[t + 1:t + 1 + H]
        want = np.nan if (np.isnan(res) or len(fut) < H) else float(fut.max() > res)
        got = fr["is_breakout"].iloc[t]
        assert (np.isnan(want) and np.isnan(got)) or want == got, (t, want, got)


def test_features_are_scale_free():
    g = make_spine(("AAA.JK",), days=700, seed=5)
    a = lf.symbol_frame(g, with_label=True)
    b = lf.symbol_frame(g.assign(Close=g["Close"] * 37.0), with_label=True)             # same stock, quoted 37x higher
    for c in config.LSTM_FEATURES + ["is_breakout"]:
        assert np.allclose(a[c].to_numpy(), b[c].to_numpy(), equal_nan=True), f"{c} depends on the price level"


def test_features_use_no_future_data():
    g = make_spine(("AAA.JK",), days=900, seed=7)
    full = lf.symbol_frame(g)
    k = 500
    cut_date = full["Date"].iloc[k - 1]
    part = lf.symbol_frame(g[g["Date"] <= cut_date])
    assert len(part) == k
    # Make the FUTURE very different: a smooth run-up to 2.2x, then a slide to 0.4x (every step stays far below the 50%
    # discontinuity threshold, otherwise chain_link would treat it as a split and rescale the whole history). Anything
    # that peeks ahead -- a centred window, a global statistic, a shifted label -- would change the features of the
    # earlier rows; a causal feature cannot.
    later = (g["Date"] > cut_date).to_numpy()
    n_later = int(later.sum())
    log_shape = np.concatenate([np.linspace(0, np.log(2.2), 40), np.linspace(np.log(2.2), np.log(0.4), 60), np.full(max(n_later - 100, 0), np.log(0.4))])[:n_later]
    shape = np.exp(log_shape)
    g2 = g.copy()
    g2.loc[later, "Close"] = g.loc[later, "Close"].to_numpy() * shape
    altered = lf.symbol_frame(g2)
    for c in config.LSTM_FEATURES:
        assert np.allclose(full[c].iloc[:k].to_numpy(), part[c].to_numpy(), equal_nan=True), f"{c} changed when later data was added"
        assert np.allclose(altered[c].iloc[:k].to_numpy(), part[c].to_numpy(), equal_nan=True), f"{c} reacts to FUTURE prices"
    for c in ("dist_res_sigma", "dist_sup_sigma"):                                    # (the altered future really is different: new high AND new low)
        assert not np.allclose(altered[c].iloc[k:].to_numpy(), full[c].iloc[k:].to_numpy(), equal_nan=True)
    # ...it breaks the rolling high AND the rolling low that were in force at the cut
    assert altered["close_adj"].iloc[k:].max() > full["yearly_resistance"].iloc[k - 1] and altered["close_adj"].iloc[k:].min() < full["yearly_support"].iloc[k - 1]


def test_weekend_rows_do_not_change_the_features():
    g = make_spine(("AAA.JK",), days=600, seed=9)
    with_cal = lf.symbol_frame(g)
    sessions_only = lf.symbol_frame(g[g["Volume"] > 0])
    for c in config.LSTM_FEATURES:
        assert np.allclose(with_cal[c].to_numpy(), sessions_only[c].to_numpy(), equal_nan=True)


def test_samples_are_consecutive_sessions_ending_at_the_labelled_row():
    fr = lf.symbol_frame(make_spine(("AAA.JK",), days=700, seed=11), with_label=True)
    smp = lf.build_samples(fr)
    L, feats = config.LSTM_SEQ_LENGTH, config.LSTM_FEATURES
    assert smp["X"].shape[1:] == (L, len(feats)) and len(smp["y"]) > 100
    for i in (0, 7, len(smp["y"]) - 1):
        e = smp["end"][i]
        assert np.allclose(smp["X"][i], fr[feats].iloc[e - L + 1:e + 1].to_numpy())
        assert smp["y"][i] == fr["is_breakout"].iloc[e] and smp["date"][i] == fr["Date"].iloc[e].to_datetime64()
        assert smp["label_end"][i] == fr["Date"].iloc[e + config.LSTM_HORIZON].to_datetime64()
    assert not np.isnan(smp["X"]).any() and not np.isnan(smp["y"]).any()


def test_split_blocks_are_ordered_disjoint_and_purged():
    g = make_spine(days=1500, seed=13)
    samples = {s: lf.build_samples(lf.symbol_frame(gg, with_label=True)) for s, gg in g.groupby("symbol")}
    date = np.concatenate([s["date"] for s in samples.values()])
    label_end = np.concatenate([s["label_end"] for s in samples.values()])
    cuts = lf.split_cutoffs(date, 0.6, 0.2, 7)
    m = lf.block_masks(date, cuts, label_end)
    assert not (m["train"] & m["val"]).any() and not (m["val"] & m["test"]).any() and not (m["train"] & m["test"]).any()
    assert date[m["train"]].max() < date[m["val"]].min() and date[m["val"]].max() < date[m["test"]].min()
    # PURGE: no label ever looks at a price from a later block
    assert pd.Timestamp(label_end[m["train"]].max()) <= cuts["d1"]
    assert pd.Timestamp(label_end[m["val"]].max()) <= cuts["d2"]
    # ...and without the purge the same cut-offs WOULD leak (so the test is meaningful)
    naive = lf.block_masks(date, cuts)
    assert pd.Timestamp(label_end[naive["train"]].max()) > cuts["d1"]
    assert m["train"].sum() < naive["train"].sum()


# ======================================================================================
# Calibration, metrics, the logistic model
# ======================================================================================
def test_platt_scaling_recovers_a_known_calibration():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 20000)
    y = (rng.random(20000) < lf.sigmoid(2.0 * z - 1.0)).astype(int)
    p = lf.fit_platt(z, y, C=100.0)
    assert p["fitted"] and abs(p["a"] - 2.0) < 0.15 and abs(p["b"] + 1.0) < 0.15


def test_platt_is_the_identity_when_there_is_only_one_class():
    p = lf.fit_platt(np.array([0.1, 0.5, 0.9]), np.zeros(3))
    assert p == {"a": 1.0, "b": 0.0, "fitted": False}
    assert np.allclose(lf.apply_platt(np.array([0.0, 2.0]), p), lf.sigmoid(np.array([0.0, 2.0])))


def test_binary_metrics_known_values():
    y = np.array([0, 0, 1, 1])
    m = lf.binary_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["auc"] == 1.0 and abs(m["brier"] - np.mean([0.01, 0.04, 0.04, 0.01])) < 1e-9 and m["n_pos"] == 2
    one = lf.binary_metrics(np.zeros(5), np.full(5, 0.1))
    assert np.isnan(one["auc"]) and abs(one["brier"] - 0.01) < 1e-9                      # AUC is undefined with one class, Brier is not
    cal = lf.binary_metrics(np.array([0] * 90 + [1] * 10), np.full(100, 0.1))
    assert cal["ece"] < 1e-9                                                              # a perfectly calibrated constant forecast


def test_logistic_model_matches_the_manual_formula_and_is_nan_where_features_are():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(500, len(config.LSTM_FEATURES)))
    y = (rng.random(500) < lf.sigmoid(X[:, 0] - 0.5)).astype(int)
    scaler = lf.fit_scaler(X)
    model = lf.fit_logit(lf.standardize(X, scaler), y)
    params = {"features": config.LSTM_FEATURES, "scaler": scaler, "logit": model, "calibration": {"a": 1.0, "b": 0.0, "fitted": False}}
    frame = pd.DataFrame(X, columns=config.LSTM_FEATURES)
    frame.iloc[3, 1] = np.nan
    p = lf.logit_probabilities(params, frame)
    assert np.isnan(p[3]) and np.isfinite(np.delete(p, 3)).all()
    z = ((X[7] - np.array(scaler["mean"])) / np.array(scaler["scale"])) @ np.array(model["coef"]) + model["intercept"]
    assert abs(p[7] - lf.sigmoid(z)) < 1e-12


# ======================================================================================
# Trainer + predictor, end to end (logistic model: no torch needed)
# ======================================================================================
def _train_on_synthetic(tmp):
    from pipeline import lstm_hurst_train as T
    spine = make_spine(days=1500, seed=21)
    csv = os.path.join(tmp, "MCS_features.csv")
    spine.to_csv(csv, index=False)
    models = os.path.join(tmp, "models")
    lines = []
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        T.main(["--csv", csv, "--model-dir", models, "--no-lstm"])
    return spine, models, buf.getvalue()


def test_trainer_writes_a_pooled_model_the_predictor_reproduces_exactly():
    import dataclasses
    import io
    import contextlib
    from pipeline.lstm_hurst_predictor import add_breakout_probability
    with tempfile.TemporaryDirectory() as tmp:
        spine, models, log = _train_on_synthetic(tmp)
        params = lf.read_params(os.path.join(models, config.LSTM_POOLED_DIRNAME))
        assert params["feature_version"] == config.LSTM_FEATURE_VERSION and params["model_type"] == "logit"
        assert params["split"]["purged_by_label_end"] and set(params["metrics"]["test"]) >= {"logit", "logit_sigma_only", "base_rate_val"}
        assert not os.path.exists(os.path.join(models, config.LSTM_POOLED_DIRNAME, config.LSTM_MODEL_FILENAME))   # no stale LSTM weights
        with contextlib.redirect_stdout(io.StringIO()):
            out = add_breakout_probability(spine, dataclasses.replace(config.Paths(), lstm_hurst_model_path=models))
        assert {"breakout_probability", "breakout_model", "hurst_exponent", "yearly_resistance"} <= set(out.columns)
        p = out["breakout_probability"].dropna()
        assert len(p) > 1000 and p.between(0, 1).all()
        # training-time and inference-time probabilities are the SAME numbers
        for sym, gg in spine.groupby("symbol"):
            fr = lf.symbol_frame(gg, with_label=True)
            smp = lf.build_samples(fr, params["features"], params["seq_length"])
            z = lf.logit_scores(params["logit"], lf.standardize(smp["X"][:, -1, :], params["scaler"]))
            want = lf.apply_platt(z, params["calibration"])
            got = out[out.symbol == sym].set_index("Date")["breakout_probability"].reindex(pd.to_datetime(smp["date"])).to_numpy()
            assert np.allclose(want, got, atol=1e-9), sym
        # weekends/holidays carry the last session's probability
        one = out[out.symbol == "AAA.JK"].sort_values("Date").reset_index(drop=True)
        for i in range(1, len(one)):
            if one.loc[i, "Volume"] == 0 and not np.isnan(one.loc[i - 1, "breakout_probability"]):
                assert one.loc[i, "breakout_probability"] == one.loc[i - 1, "breakout_probability"]
                break


def test_predictor_skips_models_it_must_not_use():
    import dataclasses
    import io
    import contextlib
    import shutil
    from pipeline.lstm_hurst_predictor import add_breakout_probability
    with tempfile.TemporaryDirectory() as tmp:
        spine, models, _ = _train_on_synthetic(tmp)
        spine = spine[spine.symbol == "AAA.JK"].reset_index(drop=True)
        paths = dataclasses.replace(config.Paths(), lstm_hurst_model_path=models)

        def run():
            with contextlib.redirect_stdout(io.StringIO()):
                return add_breakout_probability(spine, paths)["breakout_probability"].notna().sum()

        pooled = os.path.join(models, config.LSTM_POOLED_DIRNAME, config.LSTM_PARAMS_FILENAME)
        good = json.load(open(pooled))
        assert run() > 0
        for label, change in (("old-method model (no feature_version)", {"feature_version": None}),
                              ("failed its own hold-out gate", {"beats_baseline": False}),
                              ("trained with another horizon", {"horizon": 30})):
            json.dump({**good, **change}, open(pooled, "w"))
            assert run() == 0, label
        json.dump(good, open(pooled, "w"))
        config.LSTM_USE_ONLY_IF_BEATS_BASELINE = False
        try:
            json.dump({**good, "beats_baseline": False}, open(pooled, "w"))
            assert run() > 0                                                               # the gate can be switched off
        finally:
            config.LSTM_USE_ONLY_IF_BEATS_BASELINE = True
        # a company folder of the OLD trainer never shadows the pooled model
        json.dump(good, open(pooled, "w"))
        os.makedirs(os.path.join(models, "AAA"))
        json.dump({"hidden_size": 64, "seq_length": 30}, open(os.path.join(models, "AAA", config.LSTM_PARAMS_FILENAME), "w"))
        assert run() > 0


# ======================================================================================
# XGBoost trainer helpers (need xgboost; skipped otherwise)
# ======================================================================================
def _xgb_trainer():
    try:
        import xgboost  # noqa: F401
    except ImportError:
        skip("xgboost is not installed")
    from pipeline import train_xgboost_transfer as X
    return X


def test_xgboost_trainer_splits_are_chronological_and_embargoed():
    X = _xgb_trainer()
    df = make_spine(days=1000, seed=2)
    blocks, cuts = X.time_blocks(df)
    d = df["Date"]
    tr, va, te = d[blocks["train"]], d[blocks["val"]], d[blocks["test"]]
    assert tr.max() < va.min() and va.max() < te.min()
    assert (va.min() - tr.max()).days > X.EMBARGO_DAYS and (te.min() - va.max()).days > X.EMBARGO_DAYS
    assert not (blocks["train"] & blocks["val"]).any()


def test_xgboost_trainer_metrics():
    X = _xgb_trainer()
    m = X.level_metrics([100, 200, 400], [50, 100, 200])
    assert abs(m["WMAPE_%"] - 100 * 350 / 700) < 1e-9 and abs(m["bias"] - 2.0) < 1e-9          # forecasts 2x too low -> bias 2.0
    assert X.level_metrics([100, 100], [100, 100])["WMAPE_%"] == 0.0


def test_xgboost_trainer_refuses_a_features_file_with_the_old_target():
    X = _xgb_trainer()
    with tempfile.TemporaryDirectory() as tmp:
        df = make_spine(("AAA.JK",), days=200, seed=4)
        for c in config.XGBOOST_FEATURES:
            if c not in df.columns:
                df[c] = 0.0
        df["target_volume_T_plus_1"] = df["Volume"].shift(-1).fillna(0.0)                    # tomorrow's CALENDAR day: zeros on weekends
        path = os.path.join(tmp, "f.csv")
        df.to_csv(path, index=False)
        try:
            X.load_frame(path)
            raise AssertionError("accepted the old target")
        except SystemExit as e:
            assert "OLD target" in str(e)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = skipped = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Skip as e:
            skipped += 1
            print(f"  SKIP  {name}: {e}")
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed - skipped}/{len(tests)} passed" + (f", {skipped} skipped" if skipped else ""))
    sys.exit(1 if failed else 0)
