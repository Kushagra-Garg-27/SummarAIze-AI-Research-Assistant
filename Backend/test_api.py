import requests
from bs4 import BeautifulSoup
import re
from transformers import pipeline

# -------------------------------
# API KEYS (replace with yours)
# -------------------------------
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
SEARCH_ENGINE_ID = "YOUR_SEARCH_ENGINE_ID"

# -------------------------------
# Helper: Clean text
# -------------------------------
def clean_text(text):
    text = re.sub(r"\[.*?\]", "", text)   # remove [1], [citation needed]
    text = re.sub(r"\s+", " ", text)      # normalize spaces
    return text.strip()

# -------------------------------
# Source 1: Wikipedia
# -------------------------------
def scrape_wikipedia(topic):
    try:
        topic = topic.strip().replace(" ", "_")
        url = f"https://en.wikipedia.org/wiki/{topic}"
        response = requests.get(url)
        if response.status_code != 200:
            return ""
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find("div", {"id": "mw-content-text"}).find_all("p")
        text = " ".join([clean_text(p.get_text()) for p in paragraphs if p.get_text().strip()])
        return text[:3000]
    except:
        return ""

# -------------------------------
# Source 2: arXiv
# -------------------------------
def fetch_arxiv(topic, max_results=2):
    try:
        url = f"http://export.arxiv.org/api/query?search_query=all:{topic}&start=0&max_results={max_results}"
        response = requests.get(url)
        if response.status_code != 200:
            return ""
        soup = BeautifulSoup(response.text, "xml")
        summaries = soup.find_all("summary")
        if not summaries:
            return ""
        text = " ".join([clean_text(s.text) for s in summaries])
        return text[:1500]
    except:
        return ""

# -------------------------------
# Source 3: Auto Blog Finder + Scraper
# -------------------------------
def find_blog_url(topic):
    try:
        url = f"https://www.googleapis.com/customsearch/v1?q={topic}&cx={SEARCH_ENGINE_ID}&key={GOOGLE_API_KEY}"
        response = requests.get(url)
        if response.status_code != 200:
            return None
        results = response.json().get("items", [])
        if not results:
            return None
        return results[0]["link"]
    except:
        return None

def scrape_blog(url):
    try:
        if not url: return ""
        response = requests.get(url)
        if response.status_code != 200:
            return ""
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join([clean_text(p.get_text()) for p in paragraphs if p.get_text().strip()])
        return text[:2000]
    except:
        return ""

# -------------------------------
# Summarizers
# -------------------------------
def short_summary(text):
    if not text: return ""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    result = summarizer(text, max_length=120, min_length=50, do_sample=False)
    return result[0]['summary_text']

def extended_summary(text):
    if not text: return ""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    result = summarizer(text, max_length=250, min_length=100, do_sample=False)
    return result[0]['summary_text']

# -------------------------------
# Categorizer (Rule-based)
# -------------------------------
def categorize_points(text):
    sentences = re.split(r'\. |\?|\!', text)
    features, advantages, disadvantages, applications = [], [], [], []
    for s in sentences:
        s = s.strip()
        if not s: continue
        lower = s.lower()
        if any(word in lower for word in ["feature", "property", "characteristic", "includes", "aspect"]):
            features.append(s)
        elif any(word in lower for word in ["advantage", "benefit", "strength", "pro", "positive"]):
            advantages.append(s)
        elif any(word in lower for word in ["disadvantage", "limitation", "weakness", "con", "challenge", "negative"]):
            disadvantages.append(s)
        elif any(word in lower for word in ["application", "use case", "applied", "utilized", "implementation"]):
            applications.append(s)
        else:
            features.append(s)  # default bucket
    return {
        "Features": features,
        "Advantages": advantages,
        "Disadvantages": disadvantages,
        "Applications": applications
    }

# -------------------------------
# 🚀 Multi-Source Structured Pipeline
# -------------------------------
if __name__ == "__main__":
    topic = input("Enter a research topic: ")

    print(f"\n🔎 Collecting multi-source info on: {topic}\n")

    # Collect data
    wiki_text = scrape_wikipedia(topic)
    arxiv_text = fetch_arxiv(topic)
    blog_url = find_blog_url(topic)
    blog_text = scrape_blog(blog_url) if blog_url else ""

    combined_text = (wiki_text + " " + arxiv_text + " " + blog_text).strip()[:3000]

    if not combined_text:
        print("⚠️ No data fetched. Try another topic.")
    else:
        # Short overview
        overview = short_summary(combined_text)

        # Extended summary + categorize
        extended = extended_summary(combined_text)
        categories = categorize_points(extended)

        print("\n📌 Short Overview:\n")
        print(overview)

        # Always print Features
        if categories["Features"]:
            print("\n📌 Features:\n")
            for p in categories["Features"]:
                print(f"- {p}")

        if categories["Advantages"]:
            print("\n📌 Advantages:\n")
            for p in categories["Advantages"]:
                print(f"- {p}")

        if categories["Disadvantages"]:
            print("\n📌 Disadvantages:\n")
            for p in categories["Disadvantages"]:
                print(f"- {p}")

        if categories["Applications"]:
            print("\n📌 Applications / Use Cases:\n")
            for p in categories["Applications"]:
                print(f"- {p}")

        print("\n🔗 Blog Source Used:", blog_url if blog_url else "None")
