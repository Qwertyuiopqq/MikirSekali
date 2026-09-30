#!/usr/bin/env python3
"""
sentiment_trigger.py -- DEMO: score news.txt with FinBERT and push the result to the running dashboard.

    terminal A:  python website/server.py
    terminal B:  python website/sentiment_trigger.py

Reads website/news.txt (one sentence / headline per line; blank lines and lines starting with '#' are
ignored), runs every line through the fine-tuned FinBERT, averages the scores and POSTs the result to the
server. The "Sentiment Score" badge next to "Active Focus" (Competitive Urgency page) updates by itself
within ~2 seconds -- no page reload needed.

Score per sentence = P(positive) - P(negative), so it lies in [-1, +1] (same convention as test/finBERT.py,
label order 0=Positif, 1=Netral, 2=Negatif). The daily score is the mean over all lines. An empty news.txt
means "no news" and gives 0.00 (neutral).

Needs `torch` + `transformers` (this script only -- the server does not).
Options:  --news path/to/news.txt   --model path/to/model_finetuned   --url http://127.0.0.1:8000
"""
import argparse, json, sys
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
    """Fail fast (before the slow model load) if the dashboard server is not running."""
    try:
        request.urlopen(url + "/api/sentiment", timeout=3).read()
    except error.HTTPError as e:
        sys.exit(f"[!] Server di {url} menjawab HTTP {e.code} untuk /api/sentiment. "
                 f"Kemungkinan server lama belum di-restart: hentikan lalu jalankan lagi `python website/server.py`.")
    except OSError:
        sys.exit(f"[!] Server tidak terjangkau di {url}.\n"
                 f"    Jalankan dulu di terminal lain: python website/server.py")


def push(url, payload):
    req = request.Request(url + "/api/sentiment", data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode("utf-8"))
    except error.HTTPError as e:
        sys.exit(f"[!] Server menolak data (HTTP {e.code}): {e.read().decode('utf-8', 'replace')}")


def main():
    ap = argparse.ArgumentParser(description="Score news.txt with FinBERT and push it to the dashboard (demo).")
    ap.add_argument("--news", default=str(HERE / "news.txt"), help="news file, one sentence per line (default: website/news.txt)")
    ap.add_argument("--model", default=None, help="FinBERT folder (default: finbert_model_folder in pipeline/config.py)")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="dashboard server (default: %(default)s)")
    a = ap.parse_args()
    url, news_path = a.url.rstrip("/"), Path(a.news)

    if not news_path.exists():
        sys.exit(f"[!] File berita tidak ditemukan: {news_path}")
    texts = read_news(news_path)
    check_server(url)

    if texts:
        model_dir = Path(a.model) if a.model else (ROOT / "pipeline" / config.Paths().finbert_model_folder).resolve()
        items = score_texts(texts, model_dir)
    else:
        print(f"[i] {news_path.name} tidak berisi berita -> skor 0.00 (netral / tidak ada berita)")
        items = []
    score = sum(i["score"] for i in items) / len(items) if items else 0.0

    print()
    for i in items:
        print(f"  {i['label']:<8} {i['score']:+.3f}  {i['text'][:80]}")
    print(f"\n[=] Sentiment score hari ini: {score:+.3f}  ({len(items)} berita)")

    push(url, {"score": score, "count": len(items), "items": items, "source": news_path.name})
    print(f"[ok] Terkirim ke {url} -- lihat badge 'Sentiment Score' di Competitive Urgency.")


if __name__ == "__main__":
    main()
