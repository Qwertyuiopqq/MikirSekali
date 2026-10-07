"""
test_fuzzy_system.py -- executable specification of the health-score rules.

Run from the project folder (the one containing pipeline/, website/ and testing/):

    python testing/test_fuzzy_system.py          # no extra packages needed
    python -m pytest testing/test_fuzzy_system.py -v

Each test is one sentence of the intended behaviour, so when a number on the website looks wrong,
this file tells you which rule is being violated:

  1. LSTM-Hurst x support/resistance   price HIT SUPPORT -> negative, BROKE RESISTANCE -> positive
  2. BERT sentiment                    moves the score; random mock sentiment never does; live push is per company
  3. XGBoost                           a rising forecast raises the score, a falling one lowers it
  4. The fuzzy weighted system         weights sum to 1, the breakdown adds up to the headline score
  5. config.py                         where MCS_health.csv lives
"""
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
for _m in ("xgboost", "torch", "tensorflow"):         # not needed here; lets the modules import on a machine without them
    try:
        __import__(_m)
    except Exception:
        sys.modules[_m] = types.ModuleType(_m)

from pipeline import config                                                     # noqa: E402
from pipeline.fuzzy_system import BusinessHealthFuzzySystem, xgb_applicable, forecast_norm  # noqa: E402
from pipeline.support_resistance import add_sr_events, _event_strengths         # noqa: E402

FZ = BusinessHealthFuzzySystem()
PC = config.PREDICTION_COLUMN


def approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def row(**kw):
    """One neutral Wednesday row; every test changes only the field it is about."""
    base = dict(Date=pd.Timestamp("2026-09-16"), symbol="AAA.JK", Volume=1e6, day_of_week=2,
                daily_return_pct=0.01, MA_7_Close=101.0, MA_30_Close=100.0, volatility_7d=0.02,
                daily_news_sentiment=0.0, sentiment_is_mock=False, daily_news_count=5,
                **{PC: 1e6}, xgb_applicable=True, xgb_norm=1e6,
                breakout_probability=0.5, hurst_exponent=0.60, sr_break_up=0.0, sr_hit_support=0.0)
    base.update(kw)
    return pd.DataFrame([base])


def score(df, mode="predict_breakout"):
    return float(FZ.calculate_health_score(df, mode, PC, "breakout_probability" if mode == "predict_breakout" else None).iloc[0])


# ======================================================================================
# 1. LSTM-Hurst x support / resistance
# ======================================================================================
def test_hit_support_lowers_the_score_and_more_in_a_trending_regime():
    neutral = score(row())
    trending = score(row(sr_hit_support=1.0, hurst_exponent=0.80, breakout_probability=0.0))
    random_ = score(row(sr_hit_support=1.0, hurst_exponent=0.58, breakout_probability=0.0))
    assert trending < random_ < neutral, (trending, random_, neutral)


def test_hit_support_in_a_mean_reverting_regime_is_not_penalised():
    # Hurst <= SR_HURST_MEAN_REVERTING: levels tend to hold -> persistence 0 -> no penalty
    assert approx(score(row(sr_hit_support=1.0, hurst_exponent=0.40, breakout_probability=0.0)), score(row()))


def test_a_high_breakout_probability_softens_the_support_penalty():
    low_p = score(row(sr_hit_support=1.0, hurst_exponent=0.80, breakout_probability=0.0))
    high_p = score(row(sr_hit_support=1.0, hurst_exponent=0.80, breakout_probability=0.9))
    assert low_p < high_p <= score(row())


def test_break_resistance_raises_the_score_with_probability_and_trend():
    neutral = score(row())
    assert score(row(sr_break_up=1.0, breakout_probability=0.9, hurst_exponent=0.80)) > neutral
    # no probability of a breakout, or a mean-reverting regime (breaks fail) -> no credit
    assert approx(score(row(sr_break_up=1.0, breakout_probability=0.0, hurst_exponent=0.80)), neutral)
    assert approx(score(row(sr_break_up=1.0, breakout_probability=0.9, hurst_exponent=0.40)), neutral)


def test_support_hit_and_resistance_break_are_mirror_images():
    # P = 0.5, fully trending: +0.5 for a break, -0.5 for a hit -> symmetric around the neutral score
    n = score(row())
    up = score(row(sr_break_up=1.0, breakout_probability=0.5, hurst_exponent=0.80))
    dn = score(row(sr_hit_support=1.0, breakout_probability=0.5, hurst_exponent=0.80))
    assert up > n > dn and approx(up - n, n - dn)


def test_event_strength_scales_the_effect():
    n = score(row())
    full = score(row(sr_hit_support=1.0, hurst_exponent=0.8, breakout_probability=0.0)) - n
    half = score(row(sr_hit_support=0.5, hurst_exponent=0.8, breakout_probability=0.0)) - n
    assert full < 0 and approx(half, full / 2, 1e-4)


def test_no_event_means_neutral_whatever_the_probability_is():
    base = score(row(breakout_probability=0.5, hurst_exponent=0.6))
    for p in (0.0, 0.3, 1.0):
        for h in (0.3, 0.6, 0.9):
            assert approx(score(row(breakout_probability=p, hurst_exponent=h)), base)


def test_company_without_an_lstm_model_is_scored_without_the_breakout_component():
    d = row(breakout_probability=np.nan, sr_hit_support=1.0)
    ex = FZ.explain(d, "predict_breakout", PC, "breakout_probability")
    assert pd.isna(ex["membership"]["breakout"].iloc[0]) and ex["weight"]["breakout"].iloc[0] == 0
    assert approx(ex["weight"].iloc[0].sum(), 1.0)


def _toy_lines(close, low, sup, res, touches=4):
    n = len(close)
    return _event_strengths(np.array(close, float), np.array(low, float), np.full(n, sup, float), np.full(n, res, float),
                            np.full(n, float(touches)), np.full(n, float(touches)), 4, 0.0, 0.0, 3)


def test_event_detector_break_hit_and_failed_break():
    #        day:  0      1      2      3      4     5
    close = [100.0, 101.0, 104.0, 104.5, 99.0, 99.0]      # closes above resistance 103 on day 2, falls back on day 4
    low = [99.0, 100.0, 102.0, 103.5, 98.0, 94.0]       # support 95 is reached by the low on day 5
    up, dn = _toy_lines(close, low, sup=95.0, res=103.0)
    assert np.isnan(up[0]) and up[1] == 0.0
    assert up[2] == 1.0 and up[3] == 1.0                  # broke resistance
    assert up[4] == 0.0                                  # fell back under the broken level: the break failed
    assert dn[4] == 0.0 and dn[5] == 1.0                  # hit support only on day 5


def test_event_detector_ignores_weak_levels_but_accepts_the_one_year_extreme():
    close, low = [100.0, 101.0, 104.0], [99.0, 100.0, 94.0]
    up, dn = _toy_lines(close, low, sup=95.0, res=103.0, touches=2)     # 2 touches < 4: weak lines -> no event
    assert up[2] == 0.0 and dn[2] == 0.0
    up, dn = _toy_lines(close, low, sup=95.0, res=103.0, touches=0)     # touches == 0 = 1-year high/low fallback -> counts
    assert up[2] == 1.0 and dn[2] == 1.0


def test_event_detector_has_no_look_ahead():
    rng = np.random.default_rng(3)
    n = 120
    close = 100 + np.cumsum(rng.normal(0, 1.2, n))
    low = close - rng.uniform(0.2, 1.5, n)
    df = pd.DataFrame({"Date": pd.date_range("2025-01-06", periods=n, freq="B"), "symbol": "AAA.JK", "Close": close, "Low": low,
                       "Volume": 1e6, "sr_support": pd.Series(close).rolling(10).min().shift(1).bfill() - 1,
                       "sr_resistance": pd.Series(close).rolling(10).max().shift(1).bfill() + 1,
                       "sr_support_touches": 5.0, "sr_resistance_touches": 5.0})
    full = add_sr_events(df)
    for k in (40, 77, 100):
        part = add_sr_events(df.iloc[:k])
        a, b = full.iloc[:k][["sr_break_up", "sr_hit_support"]], part[["sr_break_up", "sr_hit_support"]]
        assert np.allclose(a.fillna(-1), b.fillna(-1)), f"events up to day {k} changed when later data was added"


def test_event_columns_keep_row_order_and_do_not_depend_on_it():
    rng = np.random.default_rng(5)
    n = 90
    close = 100 + np.cumsum(rng.normal(0, 1.0, n))
    df = pd.DataFrame({"Date": pd.date_range("2025-01-06", periods=n, freq="B"), "symbol": "AAA.JK", "Close": close, "Low": close - 0.5,
                       "Volume": 1e6, "sr_support": close - 2, "sr_resistance": close + 2,
                       "sr_support_touches": 5.0, "sr_resistance_touches": 5.0})
    a = add_sr_events(df)
    b = add_sr_events(df.sample(frac=1.0, random_state=1)).reindex(df.index)
    assert (a.index == df.index).all() and np.allclose(a.sr_hit_support.fillna(-1), b.sr_hit_support.fillna(-1))


# ======================================================================================
# 2. BERT sentiment
# ======================================================================================
def test_sentiment_moves_the_health_score():
    s = [score(row(daily_news_sentiment=v), "real") for v in (-0.9, 0.0, 0.9)]
    assert s[0] < s[1] < s[2], s


def test_random_mock_sentiment_never_moves_the_score():
    a = score(row(daily_news_sentiment=+0.9, sentiment_is_mock=True), "real")
    b = score(row(daily_news_sentiment=-0.9, sentiment_is_mock=True), "real")
    assert approx(a, b)
    ex = FZ.explain(row(sentiment_is_mock=True), "real")
    assert pd.isna(ex["membership"]["sentiment"].iloc[0]) and ex["weight"]["sentiment"].iloc[0] == 0


def test_a_day_without_news_has_no_sentiment_signal():
    # enrichment.py fills days without news with sentiment 0 / count 0: that is "no evidence", not a neutral reading
    ex = FZ.explain(row(daily_news_sentiment=0.0, daily_news_count=0), "real")
    assert pd.isna(ex["membership"]["sentiment"].iloc[0]) and ex["weight"]["sentiment"].iloc[0] == 0
    ex = FZ.explain(row(daily_news_sentiment=0.0, daily_news_count=3), "real")        # a real neutral reading still counts
    assert approx(ex["membership"]["sentiment"].iloc[0], 0.5) and ex["weight"]["sentiment"].iloc[0] > 0


def test_mock_sentiment_can_be_switched_back_on_in_config():
    config.SCORE_USES_MOCK_SENTIMENT = True
    try:
        a = score(row(daily_news_sentiment=+0.9, sentiment_is_mock=True), "real")
        b = score(row(daily_news_sentiment=-0.9, sentiment_is_mock=True), "real")
        assert a > b
    finally:
        config.SCORE_USES_MOCK_SENTIMENT = False


def test_live_sentiment_is_per_company():
    sys.path.insert(0, str(ROOT / "website"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("srv", str(ROOT / "website" / "server.py"))
    srv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srv)
    store, known = srv.Sentiment(), ["GIAA.JK", "SMDR.JK"]
    store.set({"symbol": "giaa", "score": 0.7, "count": 3}, known)
    assert store.get("GIAA.JK")["count"] == 3 and store.get("SMDR.JK")["count"] == 0      # the other company is untouched
    for bad in ({"symbol": "NOPE", "score": 0.1, "count": 1}, {"score": 0.1, "count": 1}, {"symbol": "GIAA", "score": "x"}):
        try:
            store.set(bad, known)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    store.set({"symbol": "GIAA", "score": 0.7, "count": 0}, known)                          # no news -> neutral, not live
    assert store.get("GIAA.JK")["score"] == 0.0 and not store.get("GIAA.JK")["count"]


# ======================================================================================
# 3. XGBoost volume outlook
# ======================================================================================
def test_a_rising_forecast_raises_and_a_falling_forecast_lowers_the_score():
    s_up = score(row(**{PC: 4e6, "xgb_norm": 1e6}), "predict")
    s_flat = score(row(**{PC: 1e6, "xgb_norm": 1e6}), "predict")
    s_down = score(row(**{PC: 0.25e6, "xgb_norm": 1e6}), "predict")
    assert s_down < s_flat < s_up, (s_down, s_flat, s_up)


def test_xgboost_component_is_not_saturated_for_realistic_volumes():
    # The old code scored forecast / 1.0 clipped to 1: every volume in the millions gave exactly 1.0 (the same score).
    scores = {score(row(**{PC: v, "xgb_norm": 1e6}), "predict") for v in (2e6, 3e6, 5e6)}
    assert len(scores) == 3
    m = FZ.memberships(row(**{PC: 1.5e6, "xgb_norm": 1e6}), "predict", PC)["xgboost"].iloc[0]
    assert 0.5 < m < 1.0


def test_xgboost_signal_ignores_the_models_level_bias():
    # the model runs ~2x low: scaling forecast AND its norm by the same factor must not change the score
    assert approx(score(row(**{PC: 2e6, "xgb_norm": 1e6}), "predict"), score(row(**{PC: 2e3, "xgb_norm": 1e3}), "predict"))


def test_forecast_for_a_weekend_is_left_out_of_the_score():
    # Friday/Saturday rows forecast the weekend (~0); that must not look like "volume collapsing"
    fri_a = row(day_of_week=4, xgb_applicable=False, **{PC: 0.0})
    fri_b = row(day_of_week=4, xgb_applicable=False, **{PC: 9e9})
    assert approx(score(fri_a, "predict"), score(fri_b, "predict"))
    assert pd.isna(FZ.memberships(fri_a, "predict", PC)["xgboost"].iloc[0])


def test_applicable_rows():
    d = pd.DataFrame({"day_of_week": [0, 1, 2, 3, 4, 5, 6, 2], "Volume": [1, 1, 1, 1, 1, 0, 0, 0]})
    assert xgb_applicable(d).tolist() == [True, True, True, True, False, False, False, False]


def test_forecast_norm_cancels_a_weekday_dependent_bias():
    # forecasts made on Mondays run 2x higher than other days (as the real model does); nothing is actually rising
    dates = pd.date_range("2025-01-06", periods=7 * 30, freq="D")
    df = pd.DataFrame({"Date": dates, "symbol": "AAA.JK", "day_of_week": dates.dayofweek,
                       "Volume": np.where(dates.dayofweek < 5, 1e6, 0.0)})
    df[PC] = np.where(df.day_of_week == 0, 2e6, 1e6)
    ok_rows = xgb_applicable(df)
    df["xgb_norm"] = forecast_norm(df, PC, ok_rows)
    m = FZ._fuzzify_xgboost(df[PC], df["xgb_norm"], ok_rows)
    scored = ok_rows & df.xgb_norm.notna()
    assert scored.sum() > 80
    assert np.allclose(m[scored], 0.5), "a pooled norm would score every Monday as 'rising'"

    # ...while a genuine jump on one Wednesday is still detected, and only on that row
    i = df.index[df.day_of_week == 2][-3]
    df.loc[i, PC] = 6e6
    df["xgb_norm"] = forecast_norm(df, PC, ok_rows)
    m2 = FZ._fuzzify_xgboost(df[PC], df["xgb_norm"], ok_rows)
    assert m2[i] > 0.9
    assert np.allclose(m2[scored & (df.index != i)], 0.5)


# ======================================================================================
# 4. The fuzzy weighted system itself
# ======================================================================================
def test_weights_sum_to_one_in_every_mode():
    for w in (config.FUZZY_WEIGHTS_REAL, config.FUZZY_WEIGHTS_PREDICT, config.FUZZY_WEIGHTS_PREDICT_BREAKOUT):
        assert approx(sum(w.values()), 1.0), w


def test_breakdown_always_adds_up_to_the_headline_score():
    for d in (row(), row(sr_hit_support=1.0, hurst_exponent=0.9), row(sentiment_is_mock=True),
              row(breakout_probability=np.nan), row(day_of_week=4, xgb_applicable=False)):
        ex = FZ.explain(d, "predict_breakout", PC, "breakout_probability")
        assert approx(ex["points"].iloc[0].sum(), ex["score"].iloc[0], 0.011)
        assert approx(ex["weight"].iloc[0].sum(), 1.0)


def test_score_is_always_between_0_and_100():
    rng = np.random.default_rng(1)
    for _ in range(300):
        d = row(daily_return_pct=rng.normal(0, .2), volatility_7d=abs(rng.normal(0, .2)), daily_news_sentiment=rng.uniform(-1, 1),
                breakout_probability=rng.uniform(0, 1), hurst_exponent=rng.uniform(0, 1.5), sr_break_up=rng.uniform(0, 1),
                sr_hit_support=rng.uniform(0, 1), xgb_norm=rng.uniform(1e5, 1e7), **{PC: rng.uniform(0, 1e8)})
        assert 0 <= score(d) <= 100


def test_a_row_without_its_price_inputs_has_no_score():
    assert np.isnan(score(row(volatility_7d=np.nan)))


def test_old_call_signature_still_works():
    # testing/test_runner.py and website/dev_fixture/make_fixture.py call it this way
    s = FZ.calculate_health_score(row(), mode="predict", xgb_col=PC)
    assert isinstance(s, pd.Series) and len(s) == 1


# ======================================================================================
# 5. config.py: where MCS_health.csv lives
# ======================================================================================
def test_mcs_health_csv_location_is_configurable():
    import dataclasses
    p = config.Paths()
    abs_p = dataclasses.replace(p, mcs_health_csv_path=str(Path("/data/x/MCS_health.csv")))
    assert Path(abs_p.health_csv()) == Path("/data/x/MCS_health.csv").resolve()
    rel = dataclasses.replace(p, mcs_health_csv_path="out/MCS_health.csv")                  # relative -> anchored at pipeline/
    assert Path(rel.health_csv()) == (ROOT / "pipeline" / "out" / "MCS_health.csv").resolve()
    empty = dataclasses.replace(p, mcs_health_csv_path="")                                  # "" -> <output_folder>/<mcs_health_csv>
    assert Path(empty.health_csv()) == (ROOT / "pipeline" / p.output_folder / p.mcs_health_csv).resolve()


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
