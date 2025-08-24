
import os
import re
import time
import json
import threading
from typing import Dict, List, Tuple, Set
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
import feedparser
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
CORS(app)

# -------------------------------
# In-memory stores
# -------------------------------
session_data: Dict[str, Dict] = {}
SESSION_TTL_SEC = 60 * 60  # 1 hour

topic_cache: Dict[str, Dict] = {}
CACHE_TTL = 60 * 30  # 30 minutes

def cleanup_sessions():
    while True:
        now = time.time()
        # sessions
        expired = [sid for sid, v in session_data.items()
                   if (now - v.get("timestamp", now)) > SESSION_TTL_SEC]
        for sid in expired:
            session_data.pop(sid, None)
        # topic cache
        expired_topics = [t for t, v in topic_cache.items()
                          if (now - v.get("timestamp", now)) > CACHE_TTL]
        for t in expired_topics:
            topic_cache.pop(t, None)
        time.sleep(300)

threading.Thread(target=cleanup_sessions, daemon=True).start()

# -------------------------------
# Helpers: Cleaning & Parsing
# -------------------------------
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; SummarAIzeBot/1.0; +http://localhost)"
}

def clean_text(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)      # remove [1], [citation needed]
    text = re.sub(r"\s+", " ", text)         # normalize whitespace
    return text.strip()

def sent_split(text: str) -> List[str]:
    parts = re.split(r'(?<=[\.\?\!])\s+', text)
    return [p.strip() for p in parts if p and p.strip()]

def short_answer_from(text: str, max_sentences: int = 2) -> str:
    sents = sent_split(text)
    return " ".join(sents[:max_sentences]) if sents else text.strip()

def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items:
        k = it.lower()
        if k and k not in seen:
            seen.add(k)
            out.append(it)
    return out
SECTION_RULES = {
    "features": {
        "keywords": ["feature", "property", "characteristic", "component", "capability", "aspect"]
    },
    "advantages": {
        "keywords": ["advantage", "benefit", "strength", "pro", "positive"]
    },
    "disadvantages": {
        "keywords": ["disadvantage", "limitation", "weakness", "con", "challenge", "risk", "negative"]
    },
    "applications": {
        "keywords": ["application", "use case", "usecase", "applied", "utilized", "deployment", "example"]
    },
}
def sectionize(text: str) -> Dict[str, List[str]]:
    """
    Break text into categorized sections using SECTION_RULES.
    Ensures overview sentences don't leak into features/advantages/etc.
    """
    sections = {k: [] for k in SECTION_RULES.keys()}
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sent in sentences:
        lower = sent.lower()
        matched = False
        for sec, cfg in SECTION_RULES.items():
            if any(kw in lower for kw in cfg["keywords"]):
                if sent not in sections[sec]:
                    sections[sec].append(sent.strip())
                matched = True
                break

        # 🚫 IMPORTANT CHANGE:
        # Do not force unmatched (neutral) sentences into "features"
        # → prevents Overview duplication.
        # Just skip them instead.

    # Clean: remove very short junk & duplicates
    for sec in sections:
        seen = set()
        cleaned = []
        for s in sections[sec]:
            if len(s.split()) > 3 and s not in seen:
                cleaned.append(s)
                seen.add(s)
        sections[sec] = cleaned[:8]  # keep top 8 max

    return sections



# -------------------------------
# Sources: Wikipedia, arXiv, CSE + generic
# -------------------------------
def scrape_wikipedia(topic: str, max_chars=3000) -> Dict:
    try:
        title = topic.strip().replace(" ", "_")
        url = f"https://en.wikipedia.org/wiki/{title}"
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=15)
        if r.status_code != 200:
            parts = title.split("_")
            if len(parts) >= 2:
                fallback = "_".join(parts[:2])
                url = f"https://en.wikipedia.org/wiki/{fallback}"
                r = requests.get(url, headers=DEFAULT_HEADERS, timeout=15)
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
            return requests.get(u, headers=DEFAULT_HEADERS, timeout=15)

        r = query(topic)
        if r.status_code != 200 or "<entry>" not in r.text:
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
        link = entries[0].find("id").text if entries[0].find("id") else "https://arxiv.org"
        return {"text": text, "url": link}
    except Exception:
        return {"text": "", "url": "https://arxiv.org"}


# -------------------------------
# Academic Sources
# -------------------------------
def fetch_pubmed(topic: str, max_results=5, max_chars=2000) -> Dict:
    """
    Fetch abstracts from PubMed (Entrez API).
    """
    try:
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={requests.utils.quote(topic)}&retmax={max_results}&retmode=json"
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return {"text": "", "url": "https://pubmed.ncbi.nlm.nih.gov"}

        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {"text": "", "url": "https://pubmed.ncbi.nlm.nih.gov"}

        # Fetch details
        id_str = ",".join(ids)
        fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={id_str}&rettype=abstract&retmode=text"
        r2 = requests.get(fetch_url, timeout=15)
        text = clean_text(r2.text)[:max_chars]

        return {"text": text, "url": "https://pubmed.ncbi.nlm.nih.gov"}
    except Exception:
        return {"text": "", "url": "https://pubmed.ncbi.nlm.nih.gov"}


def fetch_semanticscholar(topic: str, max_results=5, max_chars=2000) -> Dict:
    """
    Fetch summaries from Semantic Scholar (official API).
    """
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={requests.utils.quote(topic)}&limit={max_results}&fields=title,abstract,url"
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return {"text": "", "url": "https://www.semanticscholar.org"}

        data = r.json().get("data", [])
        abstracts = [clean_text(p.get("abstract", "")) for p in data if p.get("abstract")]
        text = " ".join(abstracts)[:max_chars]
        url_ref = data[0].get("url") if data else "https://www.semanticscholar.org"
        return {"text": text, "url": url_ref}
    except Exception:
        return {"text": "", "url": "https://www.semanticscholar.org"}

# -------------------------------
# News Sources
# -------------------------------
def fetch_bbc_news(topic: str, max_results=5, max_chars=1500) -> Dict:
    """
    Search BBC RSS feed for relevant articles.
    """
    try:
        feed = feedparser.parse("http://feeds.bbci.co.uk/news/rss.xml")
        matches = [entry.title + " " + entry.summary for entry in feed.entries if topic.lower() in entry.title.lower()]
        text = " ".join(matches[:max_results])[:max_chars]
        return {"text": text, "url": "https://www.bbc.com/news"}
    except Exception:
        return {"text": "", "url": "https://www.bbc.com/news"}

# -------------------------------
# Tech Sources
# -------------------------------
def fetch_hackernews(topic: str, max_results=5, max_chars=1500) -> Dict:
    """
    Fetch Hacker News posts using Algolia API.
    """
    try:
        url = f"https://hn.algolia.com/api/v1/search?query={requests.utils.quote(topic)}&hitsPerPage={max_results}"
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return {"text": "", "url": "https://news.ycombinator.com"}

        hits = r.json().get("hits", [])
        snippets = [h.get("title", "") + " " + str(h.get("url", "")) for h in hits]
        text = " ".join(snippets)[:max_chars]
        return {"text": text, "url": "https://news.ycombinator.com"}
    except Exception:
        return {"text": "", "url": "https://news.ycombinator.com"}


def fetch_stackoverflow(topic: str, max_results=5, max_chars=1500) -> Dict:
    """
    Fetch StackOverflow Q&A via official API.
    """
    try:
        url = f"https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance&q={requests.utils.quote(topic)}&site=stackoverflow&pagesize={max_results}"
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            return {"text": "", "url": "https://stackoverflow.com"}

        items = r.json().get("items", [])
        snippets = [i.get("title", "") for i in items]
        text = " ".join(snippets)[:max_chars]
        url_ref = items[0].get("link") if items else "https://stackoverflow.com"
        return {"text": text, "url": url_ref}
    except Exception:
        return {"text": "", "url": "https://stackoverflow.com"}

def google_cse_search(query: str, num=5) -> List[Dict]:
    try:
        url = (
            "https://www.googleapis.com/customsearch/v1"
            f"?q={requests.utils.quote(query)}"
            f"&cx={SEARCH_ENGINE_ID}&key={GOOGLE_API_KEY}"
            f"&num={num}"
        )
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=15)
        if r.status_code != 200:
            return []
        items = r.json().get("items", [])
        return [{"title": it.get("title", ""), "link": it.get("link", ""), "snippet": it.get("snippet", "")}
                for it in items]
    except Exception:
        return []

def scrape_generic(url: str, max_chars=2000) -> Dict:
    try:
        r = requests.get(url, headers=DEFAULT_HEADERS, timeout=15)
        if r.status_code != 200:
            return {"text": "", "url": url}
        soup = BeautifulSoup(r.text, "html.parser")
        paras = soup.find_all("p")
        text = " ".join([clean_text(p.get_text()) for p in paras if p.get_text().strip()])
        return {"text": text[:max_chars], "url": url}
    except Exception:
        return {"text": "", "url": url}

def find_and_scrape_blog(topic: str) -> Dict:
    # prefer explanatory resources
    results = google_cse_search(topic + " overview OR guide OR introduction site:medium.com OR site:towardsdatascience.com OR site:ibm.com OR site:mit.edu")
    if not results:
        results = google_cse_search(topic + " overview OR guide OR introduction")
        if not results:
            return {"text": "", "url": ""}
    return scrape_generic(results[0]["link"])

# -------------------------------
# Async collector
# -------------------------------
def collect_sources(topic: str, selected: set, max_chars=3000):
    """
    Fetch selected sources in parallel. Returns (chunks, sources_used).
    """
    tasks = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        if "wikipedia" in selected:
            tasks.append(("Wikipedia", executor.submit(scrape_wikipedia, topic, max_chars)))
        if "arxiv" in selected:
            tasks.append(("arXiv", executor.submit(fetch_arxiv, topic, max_chars // 2)))
        if "blog" in selected:
            tasks.append(("Blog", executor.submit(find_and_scrape_blog, topic)))

        # New integrations
        if "pubmed" in selected:
            tasks.append(("PubMed", executor.submit(fetch_pubmed, topic)))
        if "semanticscholar" in selected:
            tasks.append(("Semantic Scholar", executor.submit(fetch_semanticscholar, topic)))
        if "bbc" in selected:
            tasks.append(("BBC", executor.submit(fetch_bbc_news, topic)))
        if "hackernews" in selected:
            tasks.append(("Hacker News", executor.submit(fetch_hackernews, topic)))
        if "stackoverflow" in selected:
            tasks.append(("StackOverflow", executor.submit(fetch_stackoverflow, topic)))

        chunks, sources_used = [], []
        for name, fut in tasks:
            try:
                res = fut.result()
                if res and res.get("text"):
                    chunks.append(res["text"])
                    sources_used.append({"name": name, "url": res.get("url", "")})
            except Exception as e:
                print(f"[collect_sources] {name} error: {e}")
    return chunks, sources_used


# -------------------------------
# Models (lazy load)
# -------------------------------
_summarizer = None
_qa = None
_model_lock = threading.Lock()

def load_models():
    global _summarizer, _qa
    with _model_lock:
        if _summarizer is None:
            from transformers import pipeline
            _summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device_map="auto")
        if _qa is None:
            from transformers import pipeline
            _qa = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad", device_map="auto")

def summarize(text: str, max_len=180, min_len=60) -> str:
    if not text.strip():
        return ""
    load_models()
    out = _summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
    return out[0]["summary_text"]
def is_greeting_or_smalltalk(query: str) -> bool:
    q = query.lower().strip()
    greetings = {"hi", "hello", "hey", "yo", "sup", "good morning", "good evening", "good afternoon"}
    if q in greetings or any(q.startswith(g) for g in greetings):
        return True
    return False
# -------------------------------
# Guideline logic: intent → sections
# -------------------------------
def infer_intent(query: str) -> Set[str]:
    """Map the user's query to which sections to produce."""
    q = query.lower()

    # Self queries
    if any(k in q for k in ["who are you", "yourself", "what can you do", "tell me about yourself"]):
        return {"overview", "features", "advantages", "disadvantages", "applications"}

    # Very short definition / 'what is' / 'define'
    if re.search(r"^(what is|what's|define|definition of)\b", q) or "short definition" in q:
        return {"overview"}

    # Single-section asks
    if "feature" in q:
        return {"features"}
    if any(k in q for k in ["advantage", "pros", "benefit"]):
        return {"advantages"}
    if any(k in q for k in ["disadvantage", "cons", "limitations", "drawback", "challenges"]):
        return {"disadvantages"}
    if any(k in q for k in ["use case", "usecase", "applications", "application", "examples"]):
        return {"applications"}

    # Broad topic → richer structure
    if any(k in q for k in ["tell me about", "overview", "about", "intro", "explain", "explanation"]):
        return {"overview", "features", "advantages", "disadvantages", "applications"}

    # Default (broad) if nothing matched
    return {"overview", "features", "advantages", "disadvantages", "applications"}



# -------------------------------
# Cache helpers
# -------------------------------
def get_cached_topic(topic: str):
    entry = topic_cache.get(topic)
    if entry and (time.time() - entry.get("timestamp", 0)) < CACHE_TTL:
        return entry["data"]
    return None

def save_topic_cache(topic: str, data: Dict):
    topic_cache[topic] = {"data": data, "timestamp": time.time()}

SELF_QUERIES = [
    "who are you",
    "tell me about yourself",
    "what can you do",
    "introduce yourself",
    "about you"
]

def is_self_query(text: str) -> bool:
    t = text.lower()
    return any(q in t for q in SELF_QUERIES)

# -------------------------------
# API: summarize (guideline-compliant)
# -------------------------------
@app.route("/summarize", methods=["POST"])
def summarize_endpoint():
    """
    Body:
    {
      "topic": "Quantum Computing" | full question,
      "session_id": "abc123" (optional),
      "sources": ["wikipedia","arxiv","blog"] (optional),
      "max_chars": 3000 (optional)
    }
    """
    data = request.get_json(force=True)
    query = data.get("topic", "").strip()
    if not query:
        return jsonify({"ok": False, "error": "topic is required"}), 400

        # Special case: self-introduction (don’t scrape, answer directly)
        
    if is_self_query(query):
        session_id = data.get("session_id", str(int(time.time() * 1000)))
        response_sections = {
            "overview": "I’m your AI research assistant. I help by scraping trusted sources like Wikipedia, arXiv, and blogs, then summarizing them into structured research notes.",
            "features": [
                "Answer research questions with structured responses",
                "Provide overviews, features, pros/cons, and applications",
                "Perform follow-up Q&A based on session context",
            ],
            "advantages": [
                "Provides detailed, structured insights",
                "Saves time by summarizing multiple sources",
                "Supports both short facts and in-depth answers",
            ],
            "disadvantages": [
                "Dependent on source reliability (may inherit biases)",
                "Not always perfect at categorizing points",
                "Takes time to process longer queries",
            ],
            "applications": [
                "Academic research support",
                "Student project assistance",
                "Explaining technical concepts clearly",
            ],
        }

        response = {
            "ok": True,
            "session_id": session_id,
            "topic": "About AI Assistant",
            "answer": "I’m your AI research assistant. I gather information from reliable sources and summarize it into structured insights.",
            "sections": response_sections,
            "sources": [],
            "context": ""
        }
        return jsonify(response)

    # Infer which sections to produce from the question
    wanted_sections = infer_intent(query)

    session_id = data.get("session_id", str(int(time.time() * 1000)))
    selected = set([s.lower() for s in data.get("sources", ["wikipedia", "arxiv", "blog"])])
    max_chars = int(data.get("max_chars", 3000))

    # Serve from topic cache if the *exact* query was seen
    cached = get_cached_topic(query)
    if cached:
        session_data[session_id] = {
            "topic": query,
            "combined_text": cached.get("context", ""),
            "sections": cached["sections"],
            "sources": cached["sources"],
            "timestamp": time.time()
        }
        resp = dict(cached)
        resp["session_id"] = session_id
        return jsonify(resp)

    # Async fetch
    chunks, sources_used = collect_sources(query, selected, max_chars=max_chars)
    combined_text = (" ".join(chunks)).strip()[:max_chars]
    if not combined_text:
        return jsonify({"ok": False, "error": "No data fetched for the topic. Try a simpler query."}), 404

    # Summaries
    # 1) direct short answer (1–2 sentences)
    overview_long = summarize(combined_text, max_len=180, min_len=60)
    direct = short_answer_from(overview_long, max_sentences=2)

    # 2) if broader info requested, build sectioned details from an extended summary
    extended = summarize(combined_text, max_len=360, min_len=140)
    buckets = sectionize(extended)

    # Build response per guidelines: include only requested + non-empty sections
    from typing import Dict, List, Union

    sections: Dict[str, Union[List[str], str]] = {}

    if "overview" in wanted_sections:
        sections["overview"] = direct

    if "features" in wanted_sections and buckets["features"]:
        sections["features"] = buckets["features"]
    if "advantages" in wanted_sections and buckets["advantages"]:
        sections["advantages"] = buckets["advantages"]
    if "disadvantages" in wanted_sections and buckets["disadvantages"]:
        sections["disadvantages"] = buckets["disadvantages"]
    if "applications" in wanted_sections and buckets["applications"]:
        sections["applications"] = buckets["applications"]

    # Store in session for follow-up Q&A
    session_data[session_id] = {
        "topic": query,
        "combined_text": combined_text,
        "sections": sections,
        "sources": sources_used,
        "timestamp": time.time()
    }

    response = {
        "ok": True,
        "session_id": session_id,
        "topic": query,
        # direct answer first (UI can show this as the top bubble)
        "answer": direct,
        "sections": sections,
        "sources": sources_used,
        # keep a small context copy in cache
        "context": combined_text[:1600]
    }

    save_topic_cache(query, response)
    return jsonify(response)

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
    answer = _qa({"question": question, "context": context})
    return jsonify({"ok": True, "answer": answer.get("answer", ""), "score": float(answer.get("score", 0.0))})

# -------------------------------
# API: Search (type-ahead)
# -------------------------------
@app.route("/search", methods=["GET"])
def search_endpoint():
    q = request.args.get("q", "").strip()
    num = int(request.args.get("num", 5))
    if not q:
        return jsonify({"ok": False, "error": "q is required"}), 400
    results = google_cse_search(q, num=num)
    return jsonify({"ok": True, "query": q, "results": results})

# -------------------------------
# Health check
# -------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "status": "summarAIze backend running"})

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    # Ensure GOOGLE_API_KEY and SEARCH_ENGINE_ID are set
    app.run(host="0.0.0.0", port=8000, debug=True)
