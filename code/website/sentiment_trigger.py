#!/usr/bin/env python3
"""
sentiment_trigger.py -- score company news with FinBERT; update the running dashboard AND (optionally) the daily pipeline.

    terminal A:  python website/server.py
    terminal B:  python website/sentiment_trigger.py                 # company detected from the news text
                 python website/sentiment_trigger.py --symbol GIAA   # or say which company the news is about
                 python website/sentiment_trigger.py --save-csv      # ...and keep today's score for the pipeline

Reads website/news.txt (one sentence / headline per line; blank lines and lines starting with '#' are
ignored) and runs every line through the fine-tuned FinBERT. Score per sentence = P(positive) - P(negative),
so it lies in [-1, +1] (same convention as test/finBERT.py, label order 0=Positif, 1=Netral, 2=Negatif). A
company's daily score is the mean over ITS lines.

Which company? With --symbol every line is about that company. Without it, each line is assigned to the
companies in MCS_health.csv it names (ticker like GIAA, trade name like "Garuda Indonesia", or a distinctive
first word like "Garuda"); lines that name nobody go to the company when the file is about exactly one.

What the score does
  * POSTs one result PER COMPANY to the server. The company's sentiment badge updates within ~2 s and its fuzzy
    health score is recomputed from the new sentiment (the dashboard reloads by itself, no page refresh).
  * --save-csv also upserts today's (Date, symbol, daily_news_sentiment, daily_news_count) rows into the
    pipeline's news_sentiment.csv (Paths.sentiment_scores_csv in pipeline/config.py, or --save-csv <path>).
    That file is what makes the DAILY pipeline use real BERT sentiment instead of random mock data: the next
    `python pipeline.py prepare` + `score` run merges it by (Date, symbol). Run this once a day.

Needs `torch` + `transformers` (this script only -- the server does not).
Options:  --news path/to/news.txt   --model path/to/model_finetuned   --url http://127.0.0.1:8000
          --symbol GIAA   --save-csv [path]   --date YYYY-MM-DD (day stored by --save-csv, default: today)
"""
import argparse, csv, datetime, json, re, sys
from pathlib import Path
from urllib import error, request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # folder containing pipeline/ and website/
sys.path.insert(0, str(ROOT))
from pipeline import config              # noqa: E402

LABELS = {0: "Positif", 1: "Netral", 2: "Negatif"}   # must match how the model was fine-tuned (see test/finBERT.py)
MAX_LEN, BATCH = 128, 16

if hasattr(sys.stdout, "reconfigure"):   # Windows consoles (cp1252) would crash on unusual characters in news text
    sys.stdout.reconfigure(errors="replace")


def read_news(path):
    """One item per non-empty line; '#' lines are comments. utf-8-sig so a Notepad BOM does no harm."""
    lines = (raw.strip() for raw in Path(path).read_text(encoding="utf-8-sig").splitlines())
    return [t for t in lines if t and not t.startswith("#")]


def score_texts(texts, model_dir):
    try:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        sys.exit(f"[!] Package '{e.name}' belum terpasang. Jalankan: pip install torch transformers")
    if not model_dir.exists():
        sys.exit(f"[!] Folder model FinBERT tidak ditemukan: {model_dir}\n"
                 f"    Atur di pipeline/config.py (finbert_model_folder) atau pakai --model <folder>.")
    print(f"[..] Memuat FinBERT dari {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()
    if model.config.num_labels != 3:
        sys.exit(f"[!] Model punya {model.config.num_labels} label, script ini mengharapkan 3 (positif/netral/negatif).")

    items = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        enc = tokenizer(chunk, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LEN)
        with torch.no_grad():
            probs = F.softmax(model(**enc).logits, dim=-1)
        for text, p in zip(chunk, probs):
            items.append({"text": text, "label": LABELS[int(p.argmax())], "score": float(p[0] - p[2])})
    return items


def check_server(url):
    """Fail fast (before the slow model load) if the dashboard server is not running -- or is the old, non per-company version."""
    try:
        data = json.loads(request.urlopen(url + "/api/sentiment", timeout=3).read().decode("utf-8"))
    except error.HTTPError as e:
        sys.exit(f"[!] Server di {url} menjawab HTTP {e.code} untuk /api/sentiment. "
                 f"Kemungkinan server lama belum di-restart: hentikan lalu jalankan lagi `python website/server.py`.")
    except OSError:
        sys.exit(f"[!] Server tidak terjangkau di {url}.\n"
                 f"    Jalankan dulu di terminal lain: python website/server.py")
    if not isinstance(data, dict) or "states" not in data:
        sys.exit(f"[!] Server di {url} masih versi lama (sentimen global, bukan per perusahaan). "
                 f"Hentikan lalu jalankan lagi `python website/server.py`.")


def fetch_companies(url):
    """[{symbol, short, name}] = the companies in MCS_health.csv, asked from the running server ([] if unavailable)."""
    try:
        with request.urlopen(url + "/api/dashboard", timeout=15) as r:
            return json.loads(r.read().decode("utf-8")).get("meta", {}).get("symbols", [])
    except (OSError, ValueError):
        return []


_LEGAL = re.compile(r"\b(PT|Tbk\.?|Persero|Terbuka)\b", re.I)


def _name_pattern(name):
    """'PT Blue Bird Tbk' -> regex for 'Blue Bird' / 'Bluebird' ; None when the name is too short to be distinctive."""
    words = re.sub(r"[().,]", " ", _LEGAL.sub(" ", name or "")).split()
    return re.compile(r"\b" + r"\s*".join(map(re.escape, words)) + r"\b", re.I) if len(" ".join(words)) >= 4 else None


def mentions(text, company):
    """Does the line name this company? Ticker (case-sensitive: BLOG/BIRD are also English words) or trade name."""
    short = company.get("short") or company["symbol"].replace(".JK", "")
    if re.search(r"\b" + re.escape(short) + r"\b", text):
        return True
    pat = _name_pattern(company.get("name"))
    return bool(pat and pat.search(text))


def resolve_company(arg, companies):
    """'giaa' / 'GIAA.JK' -> the exact symbol used in MCS_health.csv (needed to match the pipeline's rows)."""
    a = arg.strip().upper()
    for c in companies:
        if a in (c["symbol"].upper(), (c.get("short") or "").upper()):
            return c["symbol"]
    if companies:
        sys.exit(f"[!] Perusahaan '{arg}' tidak ada di MCS_health.csv. Pilihan: "
                 f"{', '.join(c.get('short') or c['symbol'] for c in companies)}")
    return arg                      # server gave no list; the server validates it


def assign_lines(texts, companies, forced=None):
    """{symbol: [lines]}: --symbol -> all lines; else by mention, leftovers go to the company when there is exactly one."""
    if forced:
        return {forced: list(texts)}
    groups, loose = {}, []
    for t in texts:
        hit = [c["symbol"] for c in companies if mentions(t, c)]
        for sym in hit:
            groups.setdefault(sym, []).append(t)
        if not hit:
            loose.append(t)
    if loose and len(groups) == 1:
        next(iter(groups.values())).extend(loose)       # the file is about one company: unnamed lines belong to it
    elif loose:
        print(f"[!] {len(loose)} baris tidak menyebut perusahaan mana pun dan dilewati (pakai --symbol agar semua baris dipakai).")
    return groups


def push(url, payload):
    req = request.Request(url + "/api/sentiment", data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))
    except error.HTTPError as e:
        sys.exit(f"[!] Server menolak data (HTTP {e.code}): {e.read().decode('utf-8', 'replace')}")


def check_csv_layout(path):
    """Refuse (before anything is pushed) to touch an existing file that is not in news_sentiment.csv layout."""
    path = Path(path)
    if path.exists():
        with open(path, newline="", encoding="utf-8-sig") as f:
            cols = csv.DictReader(f).fieldnames
        if not cols or not set(CSV_COLUMNS) <= set(cols):
            sys.exit(f"[!] {path} sudah ada tetapi kolomnya bukan {CSV_COLUMNS} -- tidak diubah. "
                     f"Pakai --save-csv <file lain>.")


def save_daily_csv(path, day, results):
    """
    Upsert {symbol: (score, count)} for `day` into news_sentiment.csv, the file enrichment.py merges by
    (Date, symbol). Existing rows of other days/companies and any extra columns are kept. A company with
    no news (count 0) gets NO row -- days without news are filled as neutral by the pipeline.
    """
    path = Path(path)
    check_csv_layout(path)
    fields, rows = list(CSV_COLUMNS), []
    if path.exists():
        with open(path, newline="", encoding="utf-8-sig") as f:
            rd = csv.DictReader(f)
            fields = list(rd.fieldnames)
            rows = [r for r in rd if not (str(r["Date"])[:10] == day and r["symbol"] in results)]
    for sym, (score, count) in results.items():
        if count > 0:
            rows.append({**{c: "" for c in fields}, "Date": day, "symbol": sym,
                         "daily_news_sentiment": f"{score:.6f}", "daily_news_count": count})
    rows.sort(key=lambda r: (str(r["Date"])[:10], r["symbol"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


CSV_COLUMNS = ["Date", "symbol", "daily_news_sentiment", "daily_news_count"]


def main():
    ap = argparse.ArgumentParser(description="Score company news with FinBERT; update the dashboard (and optionally the daily pipeline).")
    ap.add_argument("--news", default=str(HERE / "news.txt"), help="news file, one sentence per line (default: website/news.txt)")
    ap.add_argument("--model", default=None, help="FinBERT folder (default: finbert_model_folder in pipeline/config.py)")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="dashboard server (default: %(default)s)")
    ap.add_argument("--symbol", default=None, help="company ALL the news is about, e.g. GIAA (default: detect from the text)")
    ap.add_argument("--save-csv", nargs="?", const="", default=None, metavar="PATH",
                    help="also upsert today's score into the pipeline's news_sentiment.csv "
                         "(default path: Paths.sentiment_scores_csv in pipeline/config.py)")
    ap.add_argument("--date", default=None, help="day stored by --save-csv, YYYY-MM-DD (default: today)")
    a = ap.parse_args()
    url, news_path = a.url.rstrip("/"), Path(a.news)

    day = a.date or datetime.date.today().isoformat()
    try:
        datetime.date.fromisoformat(day)
    except ValueError:
        sys.exit(f"[!] --date harus berformat YYYY-MM-DD (dapat: {day!r})")
    if not news_path.exists():
        sys.exit(f"[!] File berita tidak ditemukan: {news_path}")
    save_path = None
    if a.save_csv is not None:
        save_path = Path(a.save_csv) if a.save_csv else Path(config.Paths().resolve(config.Paths().sentiment_scores_csv))
        check_csv_layout(save_path)                      # fail early: nothing is scored or pushed into a bad file's run
    texts = read_news(news_path)
    check_server(url)

    companies = fetch_companies(url)
    forced = resolve_company(a.symbol, companies) if a.symbol else None
    groups = assign_lines(texts, companies, forced)
    if not groups:
        if forced:                                       # empty news file + --symbol: that company has no news today
            print(f"[i] {news_path.name} tidak berisi berita -> {forced}: skor 0.00 (netral / tidak ada berita)")
            groups = {forced: []}
        elif not texts:
            sys.exit(f"[!] {news_path.name} tidak berisi berita. Tambahkan berita, atau pakai --symbol untuk mereset satu perusahaan.")
        else:
            sys.exit("[!] Tidak ada baris yang menyebut perusahaan di MCS_health.csv, jadi perusahaannya tidak bisa ditentukan.\n"
                     "    Pakai --symbol, mis. --symbol GIAA")

    unique = list(dict.fromkeys(t for ts in groups.values() for t in ts))     # each line is scored once
    if unique:
        model_dir = Path(a.model) if a.model else Path(config.Paths().resolve(config.Paths().finbert_model_folder))
        scored = {i["text"]: i for i in score_texts(unique, model_dir)}
    else:
        scored = {}

    results = {}
    for sym, ts in groups.items():
        items = [scored[t] for t in ts]
        score = sum(i["score"] for i in items) / len(items) if items else 0.0
        print(f"\n== {sym}  ({len(items)} berita)")
        for i in items:
            print(f"  {i['label']:<8} {i['score']:+.3f}  {i['text'][:80]}")
        print(f"[=] Sentiment score {sym}: {score:+.3f}")
        state = push(url, {"symbol": sym, "score": score, "count": len(items), "items": items, "source": news_path.name})
        results[state.get("symbol") or sym] = (score, len(items))
    print(f"\n[ok] Terkirim ke {url} -- badge sentimen dan health score tiap perusahaan di atas ikut berubah "
          f"(halaman Competitive Urgency memuat ulang otomatis).")

    if save_path is not None:
        save_daily_csv(save_path, day, results)
        print(f"[ok] Skor {day} disimpan ke {save_path}\n"
              f"     Jalankan `python pipeline.py prepare` lalu `python pipeline.py score` agar pipeline harian memakai sentimen ini.")
    else:
        print("[i] Tambahkan --save-csv agar skor hari ini juga masuk ke pipeline harian (news_sentiment.csv).")


if __name__ == "__main__":
    main()
