import os
import re
import json
import time
import threading
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS

# -------------------------------
# Config (ENV VARS)
# -------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID", "")

assert GOOGLE_API_KEY and SEARCH_ENGINE_ID, \
    "Please set environment variables GOOGLE_API_KEY and SEARCH_ENGINE_ID."

# -------------------------------
# Flask app
# -------------------------------
app = Flask(__name__)
CORS(app)  # allow front-end to call APIs

# -------------------------------
# Lightweight in-memory session store
# session_data = {
#   <session_id>: {
#       "topic": str,
#       "combined_text": str,
#       "sections": {...},  # last structured output
#       "sources": [{"name":..., "url":...}, ...],
#       "timestamp": float
#   }
# }
# -------------------------------
session_data: Dict[str, Dict] = {}
SESSION_TTL_SEC = 60 * 60  # 1 hour

def cleanup_sessions():
    while True:
        now = time.time()
        expired = [sid for sid, v in session_data.items()
                   if (now - v.get("timestamp", now)) > SESSION_TTL_SEC]
        for sid in expired:
            session_data.pop(sid, None)
        time.sleep(300)

threading.Thread(target=cleanup_sessions, daemon=True).start()

# -------------------------------
# Helpers: Cleaning & Parsing
# -------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)         # remove [1], [citation needed]
    text = re.sub(r"\s+", " ", text)            # normalize whitespace
    return text.strip()

def split_sentences(text: str) -> List[str]:
    # Simple sentence split (good enough for our use)
    return [s.strip() for s in re.split(r'\. |\? |\! ', text) if s.strip()]

# -------------------------------
# Sources: Wikipedia, arXiv, Generic scrape, Google CSE
# -------------------------------
def scrape_wikipedia(topic: str, max_chars=3000) -> Dict:
    try:
        title = topic.strip().replace(" ", "_")
        url = f"https://en.wikipedia.org/wiki/{title}"
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            # fallback: try first two words
            parts = title.split("_")
            if len(parts) >= 2:
                fallback = "_".join(parts[:2])
                url = f"https://en.wikipedia.org/wiki/{fallback}"
                r = requests.get(url, timeout=15)
                if r.status_code != 200:
                    return {"text": "", "url": ""}
            else:
                return {"text": "", "url": ""}

        soup = BeautifulSoup(r.text, "html.parser")
        container = soup.find("div", {"id": "mw-content-text"})
        if not container:
            return {"text": "", "url": url}
        paras = container.find_all("p")
        text = " ".join([clean_text(p.get_text()) for p in paras if p.get_text().strip()])
        return {"text": text[:max_chars], "url": url}
    except Exception:
        return {"text": "", "url": ""}

def fetch_arxiv(topic: str, max_results=2, max_chars=1500) -> Dict:
    try:
        def query(q):
            u = f"http://export.arxiv.org/api/query?search_query=all:{q}&start=0&max_results={max_results}"
            return requests.get(u, timeout=15)

        r = query(topic)
        if r.status_code != 200 or "<entry>" not in r.text:
            # fallback to first two words (more general)
            q2 = " ".join(topic.split()[:2])
            r = query(q2)

        if r.status_code != 200:
            return {"text": "", "url": "https://arxiv.org"}

        soup = BeautifulSoup(r.text, "xml")
        entries = soup.find_all("entry")
        if not entries:
            return {"text": "", "url": "https://arxiv.org"}
        summaries = [clean_text(e.find("summary").text) for e in entries if e.find("summary")]
        text = " ".join(summaries)[:max_chars]
        # choose first entry link as a canonical url
        link = entries[0].find("id").text if entries[0].find("id") else "https://arxiv.org"
        return {"text": text, "url": link}
    except Exception:
        return {"text": "", "url": "https://arxiv.org"}

def google_cse_search(query: str, num=5) -> List[Dict]:
    try:
        url = (
            "https://www.googleapis.com/customsearch/v1"
            f"?q={requests.utils.quote(query)}"
            f"&cx={SEARCH_ENGINE_ID}&key={GOOGLE_API_KEY}"
            f"&num={num}"
        )
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return []
        items = r.json().get("items", [])
        return [{"title": it.get("title", ""), "link": it.get("link", ""), "snippet": it.get("snippet", "")}
                for it in items]
    except Exception:
        return []

def scrape_generic(url: str, max_chars=2000) -> Dict:
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return {"text": "", "url": url}
        soup = BeautifulSoup(r.text, "html.parser")
        paras = soup.find_all("p")
        text = " ".join([clean_text(p.get_text()) for p in paras if p.get_text().strip()])
        return {"text": text[:max_chars], "url": url}
    except Exception:
        return {"text": "", "url": url}

def find_and_scrape_blog(topic: str) -> Dict:
    results = google_cse_search(topic + " blog article")
    if not results:
        return {"text": "", "url": ""}
    # pick first result
    return scrape_generic(results[0]["link"])

# -------------------------------
# NLP Models (lazy global load)
# -------------------------------
_summarizer_short = None
_summarizer_long = None
_qa_model = None
_model_lock = threading.Lock()

def load_models():
    global _summarizer_short, _summarizer_long, _qa_model
    with _model_lock:
        if _summarizer_short is None or _summarizer_long is None:
            from transformers import pipeline
            # Use same model, different lengths (swap to t5-small if you need speed)
            _summarizer_short = pipeline("summarization", model="facebook/bart-large-cnn", device_map="auto")
            _summarizer_long  = _summarizer_short
        if _qa_model is None:
            from transformers import pipeline
            _qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device_map="auto")

def summarize_short(text: str) -> str:
    if not text.strip():
        return ""
    load_models()
    out = _summarizer_short(text, max_length=120, min_length=50, do_sample=False)
    return out[0]["summary_text"]

def summarize_long(text: str) -> str:
    if not text.strip():
        return ""
    load_models()
    out = _summarizer_long(text, max_length=250, min_length=100, do_sample=False)
    return out[0]["summary_text"]

def categorize_points(text: str) -> Dict[str, List[str]]:
    sentences = split_sentences(text)
    features, advantages, disadvantages, applications = [], [], [], []
    for s in sentences:
        l = s.lower()
        if any(k in l for k in ["advantage", "benefit", "strength", "pro", "positive"]):
            advantages.append(s)
        elif any(k in l for k in ["disadvantage", "limitation", "weakness", "con", "challenge", "negative"]):
            disadvantages.append(s)
        elif any(k in l for k in ["application", "use case", "applied", "utilized", "implementation"]):
            applications.append(s)
        elif any(k in l for k in ["feature", "property", "characteristic", "includes", "aspect"]):
            features.append(s)
        else:
            # default bucket to features (works well for neutral statements)
            features.append(s)

    return {
        "features": features,
        "advantages": advantages,
        "disadvantages": disadvantages,
        "applications": applications
    }

# -------------------------------
# API: summarize
# -------------------------------
@app.route("/summarize", methods=["POST"])
def summarize_endpoint():
    """
    Body:
    {
      "topic": "Quantum Computing",
      "session_id": "abc123" (optional),
      "sources": ["wikipedia","arxiv","blog"] (optional),
      "max_chars": 3000 (optional)
    }
    """
    data = request.get_json(force=True)
    topic = data.get("topic", "").strip()
    session_id = data.get("session_id", str(int(time.time()*1000)))
    selected = set([s.lower() for s in data.get("sources", ["wikipedia", "arxiv", "blog"])])
    max_chars = int(data.get("max_chars", 3000))

    if not topic:
        return jsonify({"ok": False, "error": "topic is required"}), 400

    # Collect sources
    sources_used = []
    chunks = []

    if "wikipedia" in selected:
        w = scrape_wikipedia(topic, max_chars=max_chars)
        if w["text"]:
            chunks.append(w["text"])
            sources_used.append({"name": "Wikipedia", "url": w["url"]})

    if "arxiv" in selected:
        a = fetch_arxiv(topic, max_chars=max_chars//2)
        if a["text"]:
            chunks.append(a["text"])
            sources_used.append({"name": "arXiv", "url": a["url"]})

    if "blog" in selected:
        b = find_and_scrape_blog(topic)
        if b["text"]:
            chunks.append(b["text"])
            sources_used.append({"name": "Blog", "url": b["url"]})

    combined_text = (" ".join(chunks)).strip()[:max_chars]

    if not combined_text:
        return jsonify({"ok": False, "error": "No data fetched for the topic. Try a simpler query."}), 404

    # Summarize & Structure
    overview = summarize_short(combined_text)
    extended = summarize_long(combined_text)
    cats = categorize_points(extended)

    # Build response with conditional sections (no empty sections)
    response_sections = {
        "overview": overview,
        "features": cats["features"] if cats["features"] else [],
        "advantages": cats["advantages"] if cats["advantages"] else [],
        "disadvantages": cats["disadvantages"] if cats["disadvantages"] else [],
        "applications": cats["applications"] if cats["applications"] else [],
    }

    # Store in session for follow-up Q&A
    session_data[session_id] = {
        "topic": topic,
        "combined_text": combined_text,
        "sections": response_sections,
        "sources": sources_used,
        "timestamp": time.time()
    }

    return jsonify({
        "ok": True,
        "session_id": session_id,
        "topic": topic,
        "sections": response_sections,
        "sources": sources_used
    })

# -------------------------------
# API: Q&A over session context
# -------------------------------
@app.route("/qa", methods=["POST"])
def qa_endpoint():
    """
    Body:
    {
      "session_id": "abc123",
      "question": "What are the disadvantages?",
      "context": "optional text to override session"
    }
    """
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    question = data.get("question", "").strip()
    context = data.get("context", "")

    if not question:
        return jsonify({"ok": False, "error": "question is required"}), 400

    if not context:
        if not session_id or session_id not in session_data:
            return jsonify({"ok": False, "error": "Provide session_id or context"}), 400
        context = session_data[session_id]["combined_text"]

    load_models()
    answer = _qa_model({"question": question, "context": context})
    return jsonify({"ok": True, "answer": answer.get("answer", ""), "score": float(answer.get("score", 0.0))})

# -------------------------------
# API: Search (recommendations / type-ahead)
# -------------------------------
@app.route("/search", methods=["GET"])
def search_endpoint():
    """
    Query params:
      q: query string (partial allowed)
      num: number of results (default 5)
    """
    q = request.args.get("q", "").strip()
    num = int(request.args.get("num", 5))
    if not q:
        return jsonify({"ok": False, "error": "q is required"}), 400

    results = google_cse_search(q, num=num)
    # Return lightweight objects for UI suggestions/list
    return jsonify({
        "ok": True,
        "query": q,
        "results": results
    })

# -------------------------------
# Health check
# -------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "status": "summarAIze backend running"})


if __name__ == "__main__":
    # For local dev: export GOOGLE_API_KEY and SEARCH_ENGINE_ID first.
    # On production, use a proper WSGI server.
    app.run(host="0.0.0.0", port=8000, debug=True)
