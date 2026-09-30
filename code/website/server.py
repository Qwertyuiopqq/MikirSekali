#!/usr/bin/env python3
"""
server.py -- local server: serves the UI and reads MCS_health.csv (output of `pipeline.py score`).

    python website/server.py                      # uses the path from pipeline/config.py
    python website/server.py --csv path/to/MCS_health.csv --port 8000

Only needs pandas + numpy (already required by the pipeline). No extra installs.

Sentiment badge (Competitive Urgency > Active Focus): shows the real daily_news_sentiment from the CSV per company;
while a demo is running, run `python website/sentiment_trigger.py`
in a second terminal: it scores news.txt with FinBERT and POSTs the result to /api/sentiment, which then overrides the CSV score (LIVE).
Score breakdowns reuse pipeline.fuzzy_system, so they always match the backend's own formulas.
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

REQUIRED = ["Date", "symbol", "Close", "daily_return_pct", "MA_7_Close", "MA_30_Close",
            "volatility_7d", "daily_news_sentiment", config.PREDICTION_COLUMN, "health_score_predict"]
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
    for f in [ROOT / "pipeline" / p.fundamental_json,
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
            self.df, self.mtime = df.sort_values(["symbol", "Date"]).reset_index(drop=True), m
        return self.df

    def info(self, s):
        return self.names.get(s.replace(".JK", ""), {})


class Sentiment:
    """DEMO ONLY: latest daily news sentiment, pushed by website/sentiment_trigger.py.

    Held in memory on purpose -- restarting the server resets it to "no news" (score 0),
    so a demo always starts from a clean state.
    """
    EMPTY = {"score": 0.0, "count": 0, "items": [], "source": None, "updated": None}

    def __init__(self):
        self.lock, self.state = threading.Lock(), dict(self.EMPTY)

    def get(self):
        with self.lock:
            return dict(self.state)

    def set(self, body):
        if not isinstance(body, dict):
            raise ValueError("body must be a JSON object")
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
        state = {"score": float(np.clip(score, -1, 1)) if count else 0.0,   # no news -> neutral
                 "count": count, "items": items, "source": str(body.get("source") or "")[:200] or None,
                 "updated": time.time()}
        with self.lock:
            self.state = state
        return dict(state)


def latest_sentiment(g):
    """Most recent real news sentiment for one company (from MCS_health.csv): score, article count, date."""
    if "daily_news_sentiment" not in g:
        return {"score": None, "count": None, "date": None}
    ok = g.dropna(subset=["daily_news_sentiment"])
    if ok.empty:
        return {"score": None, "count": None, "date": None}
    r = ok.iloc[-1]
    return {"score": float(r["daily_news_sentiment"]), "count": val(r, "daily_news_count"), "date": r["Date"]}


def factor_rows(r):
    pc, hb = config.PREDICTION_COLUMN, val(r, "breakout_probability")
    w = config.FUZZY_WEIGHTS_PREDICT_BREAKOUT if hb is not None else config.FUZZY_WEIGHTS_PREDICT
    sent, ret, vol = val(r, "daily_news_sentiment"), val(r, "daily_return_pct"), val(r, "volatility_7d")
    pred, act = val(r, pc), val(r, "Volume")
    m7, m30 = val(r, "MA_7_Close"), val(r, "MA_30_Close")
    f = [
        ("sentiment", "News Sentiment", Fz._fuzzify_sentiment(sent) if sent is not None else None,
         f"Sentiment {sent:+.2f} across {int(val(r, 'daily_news_count') or 0)} articles" if sent is not None else "No sentiment data"),
        ("trend", "Price Trend", Fz._fuzzify_trend(ret, m7, m30) if None not in (ret, m7, m30) else None,
         f"Daily return {ret:+.2%}; MA7 {'above' if m7 > m30 else 'below'} MA30" if None not in (ret, m7, m30) else "Insufficient price history"),
        ("stability", "Price Stability", Fz._fuzzify_stability(vol) if vol is not None else None,
         f"7-day volatility {vol:.2%} (ideal limit 5%)" if vol is not None else "No volatility data"),
        ("xgboost", "Volume Forecast (XGBoost)", Fz._fuzzify_xgboost(pred) if pred is not None else None,
         f"Predicted T+1 volume {pred:,.0f} vs latest actual {act or 0:,.0f}" if pred is not None else "No prediction"),
    ]
    if hb is not None:
        f.append(("breakout", "Breakout (LSTM-Hurst)", float(np.clip(hb, 0, 1)), f"P(Close > 1Y resistance within 30d) = {hb:.0%}"))
    out = []
    for k, label, sc, ev in f:
        sc = None if sc is None else float(sc)
        out.append({"key": k, "label": label, "score": None if sc is None else sc * 100, "weight": w[k],
                    "points": None if sc is None else sc * w[k] * 100, "evidence": ev})
    return out


def dashboard(store, symbol):
    df = store.load()
    t = df[df["Volume"] > 0] if "Volume" in df.columns and (df["Volume"] > 0).any() else df
    last = {s: g.iloc[-1] for s, g in t.groupby("symbol")}
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
                      "urgency": urg, "sentiment": latest_sentiment(t[t.symbol == s]), "regime": val(r, "sr_regime"), "breakout": val(r, "breakout_probability"),
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
    fs = factor_rows(r)
    stab = next((f["score"] for f in fs if f["key"] == "stability"), None)
    focus = {"symbol": symbol, "short": symbol.replace(".JK", ""), **store.info(symbol), "date": r["Date"], "close": val(r, "Close"),
             "ret": val(r, "daily_return_pct"), "health_predict": hp, "health_real": hr,
             "drift30": None if hp is None or not len(old) else float(hp - old.iloc[-1]),
             "stance": stance(hp), "stance_real": stance(hr), "factors": fs,
             "coverage": float(recent[cov_cols].notna().mean().mean()) if cov_cols else None,
             "fit": None if mape is None else float(max(0, 1 - mape)), "stability": stab,
             "breakout": val(r, "breakout_probability"), "hurst": val(r, "hurst_exponent"), "regime": val(r, "sr_regime"),
             "sr": {k: val(r, k) for k in ["sr_support", "sr_resistance", "sr_support_touches", "sr_resistance_touches",
                                           "sr_dist_to_support_pct", "sr_dist_to_resistance_pct"]},
             "volume": {"actual": val(r, "Volume"), "pred": val(r, config.PREDICTION_COLUMN), "ma7": val(r, "MA_7_Volume")},
             "sentiment": latest_sentiment(g),
             "dividend": bool(val(r, "is_dividend")), "momentum": mom(symbol)}
    pick = [symbol] + [p["symbol"] for p in peers[:3]]
    lines = []
    for s in pick:
        v = piv_c[s].dropna()
        lines.append({"symbol": s, "short": s.replace(".JK", ""), "values": (v / v.iloc[0] * 100).reindex(piv_c.index).tolist() if len(v) else []})
    return {"meta": {"csv": str(store.csv), "mtime": store.mtime, "asof": df["Date"].max(), "rows": len(df),
                     "symbols": [{"symbol": s, "short": s.replace(".JK", ""), **store.info(s), "health": val(last[s], "health_score_predict")} for s in syms],
                     "engine": "XGB + LSTM + Fuzzy"},
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
                    return self.send_json(dashboard(store, parse_qs(u.query).get("symbol", [""])[0]))
                if u.path == "/api/status":
                    store.load()
                    return self.send_json({"mtime": store.mtime, "csv": str(store.csv)})
                if u.path == "/api/sentiment":
                    return self.send_json(sentiment.get())
            except FileNotFoundError as e:
                return self.send_json({"error": str(e), "hint": "Run `python pipeline.py score` (inside pipeline/) or start the server with --csv <path>."}, 503)
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
                return self.send_json(sentiment.set(json.loads(self.rfile.read(n).decode("utf-8"))))
            except ValueError as e:            # bad JSON / bad fields (JSONDecodeError is a ValueError)
                return self.send_json({"error": str(e)}, 400)
            except Exception as e:
                return self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()
    return H


def default_csv():
    p = config.Paths()
    for c in [(ROOT / "pipeline" / p.output_folder / p.mcs_health_csv).resolve(), ROOT / "data" / "output" / p.mcs_health_csv]:
        if c.exists():
            return c
    return (ROOT / "pipeline" / p.output_folder / p.mcs_health_csv).resolve()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None, help="path to MCS_health.csv (default: from pipeline/config.py)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    store = Store(a.csv or default_csv())
    print(f"CSV   : {store.csv}  [{'found' if store.csv.exists() else 'NOT FOUND - run `python pipeline.py score` first'}]")
    print(f"Open  : http://{a.host}:{a.port}/   (Ctrl+C to stop)")
    print(f"Demo  : in a 2nd terminal run `python website/sentiment_trigger.py` to push news.txt sentiment to the dashboard")
    ThreadingHTTPServer((a.host, a.port), make_handler(store, Sentiment())).serve_forever()
