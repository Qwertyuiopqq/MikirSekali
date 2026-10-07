#!/usr/bin/env python3
"""
server.py -- local server: serves the UI and reads MCS_health.csv (output of `pipeline.py score`).

    python website/server.py                      # uses Paths.mcs_health_csv_path from pipeline/config.py
    python website/server.py --csv path/to/MCS_health.csv --port 8000

Only needs pandas + numpy (already required by the pipeline). No extra installs.

Where the numbers come from
---------------------------
  * Every price/volume/forecast/S-R/Hurst/LSTM number is read from MCS_health.csv.
  * The HEALTH SCORES are re-derived on load from that file's component columns with pipeline.fuzzy_system
    (the same code `pipeline.py score` uses), so the website always reflects the current formulas even if
    the CSV was written by an older version. Nothing on the dashboard is random or hard-coded.
  * Sentiment: the CSV's daily_news_sentiment is only used when it is REAL (not the pipeline's random mock
    fallback). A LIVE company sentiment pushed by `python website/sentiment_trigger.py` (FinBERT over the
    news file, POSTed to /api/sentiment) replaces today's sentiment of THAT company and the fuzzy health
    score is recomputed from it, so the dashboard moves when the news does.
"""
import argparse, json, sys, threading, time
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # folder containing pipeline/ and website/
WEB = HERE / "frontend" / "design"
sys.path.insert(0, str(ROOT))
from pipeline import config                                        # noqa: E402
from pipeline.fuzzy_system import BusinessHealthFuzzySystem as Fz  # noqa: E402
from pipeline.fuzzy_system import FORECAST_COLUMNS                 # noqa: E402
from pipeline.support_resistance import SR_EVENT_COLUMNS           # noqa: E402

FZ = Fz()
# What the scoring needs from the CSV. The health_score_* columns are NOT required any more: they are recomputed.
REQUIRED = ["Date", "symbol", "Close", "Volume", "daily_return_pct", "MA_7_Close", "MA_30_Close",
            "volatility_7d", "daily_news_sentiment", config.PREDICTION_COLUMN]
STANCES = [(20, "AVOID"), (40, "WATCH"), (60, "EXPLORE"), (80, "STRONG CASE"), (101, "HIGH CONVICTION")]


def stance(score):
    return None if score is None else next(lbl for lim, lbl in STANCES if score < lim)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, pd.Timestamp):
        return o.strftime("%Y-%m-%d")
    return o


def val(r, k):
    return None if k not in r.index or pd.isna(r[k]) else r[k]


def load_names():
    p = config.Paths()
    for f in [Path(p.resolve(p.fundamental_json)),
              ROOT / "testing/sample_data/real_data_raw/top10-transportation-by-marketcap-company_report.json"]:
        try:
            ov = json.load(open(f, encoding="utf-8"))["overview"]
            return {str(o["symbol"]).replace(".JK", ""): {"name": o.get("company_name"), "sector": o.get("sub_sector") or o.get("sector")} for o in ov}
        except Exception:
            continue
    return {}


class Store:
    def __init__(self, csv):
        self.csv, self.mtime, self.df, self.names = Path(csv), None, None, load_names()

    def load(self):
        if not self.csv.exists():
            raise FileNotFoundError(f"MCS_health.csv not found at {self.csv}")
        m = self.csv.stat().st_mtime
        if m != self.mtime:
            df = pd.read_csv(self.csv, parse_dates=["Date"])
            miss = [c for c in REQUIRED if c not in df.columns]
            if miss:
                raise ValueError(f"{self.csv.name} is missing columns: {', '.join(miss)}. Run `python pipeline.py score` first.")
            df = df.sort_values(["symbol", "Date"]).reset_index(drop=True)
            self.df, self.mtime = self.rescore(df), m
        return self.df

    def rescore(self, df):
        """
        Score every row with the CURRENT fuzzy formulas (pipeline.fuzzy_system) from the CSV's component
        columns. The helper columns (forecast norm, S/R events) are rebuilt too, so a change in config.py
        takes effect on restart without re-running the pipeline.
        """
        if "sentiment_is_mock" not in df.columns:
            # CSV written before the flag existed: it is mock when the real sentiment file was missing.
            p = config.Paths()
            real = Path(p.resolve(p.sentiment_scores_csv)).exists()
            df["sentiment_is_mock"] = not real
            if not real:
                print("[server] WARNING: this CSV's sentiment is MOCK (random) -- it is ignored by the health score "
                      f"and shown as 'No data'. Real sentiment file not found: {p.resolve(p.sentiment_scores_csv)}")
        stored = {c: df[c].copy() for c in ("health_score_real", "health_score_predict") if c in df.columns}
        df = df.drop(columns=[c for c in FORECAST_COLUMNS + SR_EVENT_COLUMNS if c in df.columns])
        df = FZ.score_frame(df)
        diff = sum(int(((df[c] - old).abs() > 0.011).sum()) for c, old in stored.items())
        if diff:
            print(f"[server] {self.csv.name}: health scores re-derived with the current formulas "
                  f"({diff} stored values differed -> the file was written by an older version; "
                  "run `python pipeline.py score` to refresh it).")
        return df

    def info(self, s):
        return self.names.get(s.replace(".JK", ""), {})


def resolve_symbol(raw, known):
    """'giaa' / 'GIAA.JK' -> 'GIAA.JK' if it is a company in the CSV, else None."""
    s = str(raw or "").strip().upper()
    for k in known:
        if s and s in (k.upper(), k.upper().replace(".JK", "")):
            return k
    return None


class Sentiment:
    """LIVE news sentiment PER COMPANY, pushed by website/sentiment_trigger.py (FinBERT over a news file).

    A pushed state replaces that company's sentiment for its latest trading day, and the dashboard recomputes
    the fuzzy health score from it (see with_live). Held in memory on purpose -- restarting the server resets
    every company to "no live news", so a demo always starts from a clean state. The trigger can additionally
    save the day's score into news_sentiment.csv, which is how it reaches the daily pipeline for good.
    """
    EMPTY = {"symbol": None, "score": 0.0, "count": 0, "items": [], "source": None, "updated": None}

    def __init__(self):
        self.lock, self.states = threading.Lock(), {}

    def get(self, symbol):
        with self.lock:
            return dict(self.states.get(symbol, self.EMPTY))

    def all_states(self):
        with self.lock:
            return {k: dict(v) for k, v in self.states.items()}

    def snapshot(self):
        st = self.all_states()
        return {"updated": max((v["updated"] for v in st.values()), default=None), "states": st}

    def set(self, body, known=()):
        if not isinstance(body, dict):
            raise ValueError("body must be a JSON object")
        raw = body.get("symbol")
        symbol = resolve_symbol(raw, known) if known else (str(raw).strip().upper() or None)
        if symbol is None:
            raise ValueError(f"'symbol' must name a company in MCS_health.csv (got {raw!r}); known: {', '.join(sorted(known)) or 'n/a'}")
        try:
            score = float(body.get("score"))
        except (TypeError, ValueError):
            raise ValueError("'score' must be a number between -1 and 1")
        if not np.isfinite(score):
            raise ValueError("'score' must be a finite number between -1 and 1")
        items = []
        for it in (body.get("items") or [])[:50]:
            if isinstance(it, dict):
                try:
                    sc = float(it.get("score"))
                except (TypeError, ValueError):
                    sc = None
                items.append({"text": str(it.get("text", ""))[:300], "label": str(it.get("label", ""))[:20],
                              "score": sc if sc is not None and np.isfinite(sc) else None})
        try:
            count = max(0, int(body.get("count", len(items))))
        except (TypeError, ValueError):
            raise ValueError("'count' must be an integer")
        state = {"symbol": symbol, "score": float(np.clip(score, -1, 1)) if count else 0.0,   # no news -> neutral
                 "count": count, "items": items, "source": str(body.get("source") or "")[:200] or None,
                 "updated": time.time()}
        with self.lock:
            self.states[symbol] = state
        return dict(state)


def with_live(r, st):
    """
    Latest row of one company with its LIVE sentiment swapped in and both health scores recomputed by the
    same fuzzy system (so the headline, the factor ledger and the peers list all move together).
    """
    if not st or not st.get("count"):
        return r
    row = r.to_dict()
    row.update(daily_news_sentiment=st["score"], daily_news_count=st["count"], sentiment_is_mock=False)
    return FZ.score_frame(pd.DataFrame([row])).iloc[0]


def latest_sentiment(g, live=None):
    """Sentiment to show for one company: LIVE push first, else the most recent REAL news day in the CSV, else none."""
    if live and live.get("count"):
        return {"score": live["score"], "count": live["count"], "date": time.strftime("%Y-%m-%d"),
                "live": True, "source": live.get("source"), "updated": live.get("updated")}
    none = {"score": None, "count": None, "date": None, "live": False,
            "mock": bool("sentiment_is_mock" in g and len(g) and g["sentiment_is_mock"].astype(bool).all())}
    if "daily_news_sentiment" not in g or none["mock"]:
        return none
    ok = g.dropna(subset=["daily_news_sentiment"])
    if "sentiment_is_mock" in ok:
        ok = ok[~ok["sentiment_is_mock"].astype(bool)]
    if "daily_news_count" in ok and (ok["daily_news_count"] > 0).any():
        ok = ok[ok["daily_news_count"] > 0]           # days without news are a neutral filler, not a reading
    if ok.empty:
        return none
    r = ok.iloc[-1]
    return {"score": float(r["daily_news_sentiment"]), "count": val(r, "daily_news_count"), "date": r["Date"], "live": False, "mock": False}


FACTOR_LABELS = {"sentiment": "News Sentiment", "trend": "Price Trend", "stability": "Price Stability",
                 "xgboost": "Volume Outlook (XGBoost)", "breakout": "Breakout / Support (LSTM-Hurst)"}


def factor_rows(r, live=None):
    """
    The fuzzy factor ledger for one row, taken from FZ.explain so it ALWAYS adds up to the headline score:
    `score` = membership x 100, `weight` = the weight actually used (components without a signal are left
    out and the rest renormalised), `points` = score points contributed.
    """
    pc = config.PREDICTION_COLUMN
    ex = FZ.explain(pd.DataFrame([r.to_dict()]), "predict_breakout", pc, "breakout_probability")
    mem, wt, pts = ex["membership"].iloc[0], ex["weight"].iloc[0], ex["points"].iloc[0]

    sent, ret, vol = val(r, "daily_news_sentiment"), val(r, "daily_return_pct"), val(r, "volatility_7d")
    m7, m30 = val(r, "MA_7_Close"), val(r, "MA_30_Close")
    pred, norm, applicable = val(r, pc), val(r, "xgb_norm"), bool(val(r, "xgb_applicable"))
    P, H = val(r, "breakout_probability"), val(r, "hurst_exponent")
    up, dn = val(r, "sr_break_up") or 0.0, val(r, "sr_hit_support") or 0.0

    if live and live.get("count"):
        ev_sent = f"LIVE FinBERT sentiment {sent:+.2f} over {int(live['count'])} news item(s) -- replaces today's CSV value"
    elif mem["sentiment"] != mem["sentiment"]:     # NaN: no signal
        ev_sent = ("No real news sentiment for this company (the CSV only holds random mock data) -- left out of the score"
                   if val(r, "sentiment_is_mock") else "No news for this company on this day -- left out of the score")
    else:
        ev_sent = f"Sentiment {sent:+.2f} across {int(val(r, 'daily_news_count') or 0)} articles"

    if applicable and pred is not None and norm:
        d = pred / norm - 1
        ev_xgb = (f"Forecast {pred:,.0f} vs the model's usual {norm:,.0f} for this weekday ({d:+.0%}) "
                  f"-> volume expected to {'rise' if d >= 0 else 'fall'}")
    elif not applicable:
        ev_xgb = "No outlook: the next calendar day is not a trading day (the model forecasts ~0 for weekends/holidays) -- left out of the score"
    else:
        ev_xgb = "Not enough forecast history yet to judge rise/fall -- left out of the score"

    ev_brk = None
    if P is not None:
        h = None if H is None else float(Fz.hurst_persistence(H))
        reg = val(r, "sr_regime") or "unknown"
        base = f"P(Close > its 252-day high within 30d) = {P:.0%}; Hurst {H:.2f} ({reg}) -> trend persistence {h:.0%}" if H is not None else f"P = {P:.0%}"
        parts = []
        if up > 0:
            parts.append(f"price BROKE resistance (strength {up:.0%}): +P x persistence")
        if dn > 0:
            parts.append(f"price HIT support {rp_(val(r, 'sr_support'))} (strength {dn:.0%}): -(1-P) x persistence")
        ev_brk = (" | ".join(parts) + ". " if parts else "No support/resistance event -> neutral. ") + base

    evidence = {
        "sentiment": ev_sent,
        "trend": (f"Daily return {ret:+.2%}; MA7 {'above' if m7 > m30 else 'below'} MA30"
                  if None not in (ret, m7, m30) else "Insufficient price history"),
        "stability": f"7-day volatility {vol:.2%} (ideal limit 5%)" if vol is not None else "No volatility data",
        "xgboost": ev_xgb, "breakout": ev_brk,
    }
    out = []
    for k in ("sentiment", "trend", "stability", "xgboost", "breakout"):
        if k == "breakout" and P is None:
            continue                                  # no LSTM-Hurst model for this company (explained on the page)
        m = mem[k]
        out.append({"key": k, "label": FACTOR_LABELS[k], "score": None if m != m else float(m) * 100,
                    "weight": 0.0 if wt[k] != wt[k] else float(wt[k]),
                    "points": None if m != m else float(pts[k]), "evidence": evidence[k]})
    return out


def rp_(v):
    return "n/a" if v is None else f"Rp {v:,.2f}" if v < 100 else f"Rp {v:,.0f}"


def dashboard(store, symbol, live=None):
    live = live or {}
    df = store.load()
    t = df[df["Volume"] > 0] if "Volume" in df.columns and (df["Volume"] > 0).any() else df
    # latest trading row per company; a company with a LIVE sentiment push is re-scored from it
    last = {s: with_live(g.iloc[-1], live.get(s)) for s, g in t.groupby("symbol")}
    syms = sorted(last)
    if symbol not in last:
        symbol = max(syms, key=lambda s: -1 if pd.isna(last[s]["health_score_predict"]) else last[s]["health_score_predict"])
    piv_r = t.pivot_table(index="Date", columns="symbol", values="daily_return_pct").tail(90)
    piv_c = t.pivot_table(index="Date", columns="symbol", values="Close").tail(90)

    def mom(s):
        c = t.loc[t.symbol == s, "Close"]
        return None if len(c) < 6 else float(c.iloc[-1] / c.iloc[-min(31, len(c))] - 1)

    peers = []
    for s in syms:
        if s == symbol:
            continue
        r, m = last[s], mom(s)
        ov = piv_r[symbol].corr(piv_r[s]) if piv_r[symbol].notna().sum() > 4 else np.nan
        ov = None if pd.isna(ov) else float(max(ov, 0))
        h = val(r, "health_score_predict")
        urg = 100 * (0.4 * (h or 0) / 100 + 0.3 * (ov or 0) + 0.3 * float(np.clip((m or 0) / 0.2, 0, 1)))
        peers.append({"symbol": s, "short": s.replace(".JK", ""), **store.info(s), "health": h, "momentum": m, "overlap": ov,
                      "urgency": urg, "sentiment": latest_sentiment(t[t.symbol == s], live.get(s)), "regime": val(r, "sr_regime"), "breakout": val(r, "breakout_probability"),
                      "label": "Critical Threat" if urg >= 75 else "Rising Pressure" if urg >= 55 else "Monitor"})
    peers.sort(key=lambda p: -p["urgency"])

    r, g = last[symbol], t[t.symbol == symbol]
    hp, hr = val(r, "health_score_predict"), val(r, "health_score_real")
    hs = g.set_index("Date")["health_score_predict"].dropna()
    old = hs[hs.index <= hs.index[-1] - pd.Timedelta(days=30)] if len(hs) else hs
    recent = g.tail(30)
    fit_rows = g.tail(90).dropna(subset=["target_volume_T_plus_1", config.PREDICTION_COLUMN]) if "target_volume_T_plus_1" in g else g.iloc[0:0]
    fit_rows = fit_rows[fit_rows["target_volume_T_plus_1"] > 0]
    mape = float((abs(fit_rows[config.PREDICTION_COLUMN] - fit_rows["target_volume_T_plus_1"]) / fit_rows["target_volume_T_plus_1"]).mean()) if len(fit_rows) else None
    cov_cols = [c for c in ["daily_news_sentiment", "daily_return_pct", "volatility_7d", config.PREDICTION_COLUMN, "breakout_probability"] if c in g]
    cov = recent[cov_cols].copy()
    if "daily_news_sentiment" in cov and "sentiment_is_mock" in recent:     # mock noise is not coverage
        cov.loc[recent["sentiment_is_mock"].astype(bool).to_numpy(), "daily_news_sentiment"] = np.nan
    if live.get(symbol, {}).get("count") and "daily_news_sentiment" in cov and len(cov):
        cov.iloc[-1, cov.columns.get_loc("daily_news_sentiment")] = live[symbol]["score"]
    fs = factor_rows(r, live.get(symbol))
    stab = next((f["score"] for f in fs if f["key"] == "stability"), None)
    pc = config.PREDICTION_COLUMN
    applicable, pred, norm = bool(val(r, "xgb_applicable")), val(r, pc), val(r, "xgb_norm")
    focus = {"symbol": symbol, "short": symbol.replace(".JK", ""), **store.info(symbol), "date": r["Date"], "close": val(r, "Close"),
             "ret": val(r, "daily_return_pct"), "health_predict": hp, "health_real": hr,
             "drift30": None if hp is None or not len(old) else float(hp - old.iloc[-1]),
             "stance": stance(hp), "stance_real": stance(hr), "factors": fs,
             "coverage": float(cov.notna().mean().mean()) if cov_cols else None,
             "fit": None if mape is None else float(max(0, 1 - mape)), "stability": stab,
             "breakout": val(r, "breakout_probability"), "hurst": val(r, "hurst_exponent"), "regime": val(r, "sr_regime"),
             "sr": {k: val(r, k) for k in ["sr_support", "sr_resistance", "sr_support_touches", "sr_resistance_touches",
                                           "sr_dist_to_support_pct", "sr_dist_to_resistance_pct", "sr_break_up", "sr_hit_support"]},
             # `pred`/`norm`/`vs_norm` are None when the forecast is about a non-trading day (Fri/Sat rows forecast ~0).
             "volume": {"actual": val(r, "Volume"), "pred": pred if applicable else None, "norm": norm if applicable else None,
                        "vs_norm": (pred / norm - 1) if applicable and pred is not None and norm else None,
                        "applicable": applicable, "ma7": val(r, "MA_7_Volume")},
             "sentiment": latest_sentiment(g, live.get(symbol)),
             "dividend": bool(val(r, "is_dividend")), "momentum": mom(symbol)}
    pick = [symbol] + [p["symbol"] for p in peers[:3]]
    lines = []
    for s in pick:
        v = piv_c[s].dropna()
        lines.append({"symbol": s, "short": s.replace(".JK", ""), "values": (v / v.iloc[0] * 100).reindex(piv_c.index).tolist() if len(v) else []})
    return {"meta": {"csv": str(store.csv), "mtime": store.mtime, "asof": df["Date"].max(), "rows": len(df),
                     "symbols": [{"symbol": s, "short": s.replace(".JK", ""), **store.info(s), "health": val(last[s], "health_score_predict")} for s in syms],
                     "engine": "XGB + LSTM + Fuzzy", "sentiment_live": sorted(k for k, v in live.items() if v.get("count")),
                     "sentiment_mock": bool("sentiment_is_mock" in df and df["sentiment_is_mock"].astype(bool).all())},
            "focus": focus, "peers": peers, "chart": {"dates": list(piv_c.index), "lines": lines}}


def make_handler(store, sentiment):
    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(WEB), **k)

        def send_json(self, obj, code=200):
            body = json.dumps(clean(obj), allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            try:
                if u.path == "/api/dashboard":
                    return self.send_json(dashboard(store, parse_qs(u.query).get("symbol", [""])[0], sentiment.all_states()))
                if u.path == "/api/status":
                    store.load()
                    return self.send_json({"mtime": store.mtime, "csv": str(store.csv)})
                if u.path == "/api/sentiment":
                    q = parse_qs(u.query).get("symbol", [""])[0]
                    if q:                                    # one company's live state
                        return self.send_json(sentiment.get(resolve_symbol(q, store.load()["symbol"].unique()) or q))
                    return self.send_json(sentiment.snapshot())  # {"updated": ts, "states": {symbol: state}}
            except FileNotFoundError as e:
                return self.send_json({"error": str(e), "hint": "Set Paths.mcs_health_csv_path in pipeline/config.py (or start the server with --csv <path>), then run `python pipeline.py score` if the file does not exist yet."}, 503)
            except Exception as e:
                return self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)
            if u.path in ("/", ""):
                self.path = "/mikir-sekali.html"
            return super().do_GET()

        def do_POST(self):
            if urlparse(self.path).path != "/api/sentiment":
                return self.send_json({"error": "not found"}, 404)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if not 0 < n <= 1_000_000:
                    raise ValueError("request body is empty or larger than 1 MB")
                try:
                    known = list(store.load()["symbol"].unique())
                except FileNotFoundError:
                    known = []                              # no CSV yet: accept the symbol as given
                return self.send_json(sentiment.set(json.loads(self.rfile.read(n).decode("utf-8")), known))
            except ValueError as e:            # bad JSON / bad fields (JSONDecodeError is a ValueError)
                return self.send_json({"error": str(e)}, 400)
            except Exception as e:
                return self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()
    return H


def default_csv():
    """Where MCS_health.csv is: Paths.mcs_health_csv_path in pipeline/config.py (edit it there)."""
    p = config.Paths()
    configured = Path(p.health_csv())
    if configured.exists():
        return configured
    # not at the configured spot: fall back to the older guesses so an existing setup keeps working
    for c in [(ROOT / "pipeline" / p.output_folder / p.mcs_health_csv).resolve(), ROOT / "data" / "output" / p.mcs_health_csv]:
        if c.exists():
            print(f"[server] MCS_health.csv not found at the configured path {configured} -- using {c} instead. "
                  "Set Paths.mcs_health_csv_path in pipeline/config.py to silence this.")
            return c
    return configured


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None, help="path to MCS_health.csv (default: Paths.mcs_health_csv_path in pipeline/config.py)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    store = Store(a.csv or default_csv())
    print(f"CSV   : {store.csv}  [{'found' if store.csv.exists() else 'NOT FOUND - set Paths.mcs_health_csv_path in pipeline/config.py, or run `python pipeline.py score`'}]")
    if store.csv.exists():
        try:
            d = store.load()
            print(f"Data  : {len(d)} rows, {d['symbol'].nunique()} companies, up to {d['Date'].max():%Y-%m-%d}")
        except Exception as e:
            print(f"Data  : could not be loaded yet ({type(e).__name__}: {e})")
    print(f"Open  : http://{a.host}:{a.port}/   (Ctrl+C to stop)")
    print(f"Demo  : in a 2nd terminal run `python website/sentiment_trigger.py` to push FinBERT news sentiment for a company")
    ThreadingHTTPServer((a.host, a.port), make_handler(store, Sentiment())).serve_forever()
