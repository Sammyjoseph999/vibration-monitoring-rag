"""Fetch web pages and reduce them to readable text."""
import hashlib
import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

USER_AGENT = "vibration-rag/0.1 (educational project)"
TEXT_TAGS = ["h1", "h2", "h3", "p", "li"]
DROP_TAGS = ["script", "style", "nav", "footer", "header", "form", "noscript", "aside"]


def clean_text(text):
    """Collapse whitespace and strip citation markers such as [12].

    Punctuation is kept on purpose: the chunker splits on sentence endings.
    """
    text = re.sub(r"\[\s*\d+\s*\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def html_to_passages(html, min_words=5):
    """Return the readable passages of a page, in document order, without repeats."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(DROP_TAGS):
        tag.decompose()
    # Rendered formulas come through as markup fragments, not readable text
    for tag in soup.select("math, .mwe-math-element, sup.reference"):
        tag.decompose()

    passages, seen = [], set()
    for element in soup.find_all(TEXT_TAGS):
        text = clean_text(element.get_text(" ", strip=True))
        is_heading = element.name in ("h1", "h2", "h3")
        if not text or text in seen:
            continue
        if not is_heading and len(text.split()) < min_words:
            continue  # menu items, button labels and similar fragments
        seen.add(text)
        # A heading becomes a short sentence so it stays attached to what follows
        passages.append(text.rstrip(".:") + "." if is_heading else text)
    return passages


def fetch(url, cache_dir=None, timeout=20):
    """Download a page and return its passages. Uses a local cache when given one."""
    cache_file = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".json")
        if cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))

    response = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    passages = html_to_passages(response.text)

    if cache_file is not None:
        cache_file.write_text(json.dumps(passages), encoding="utf-8")
    return passages


def fetch_all(sources, cache_dir=None):
    """Fetch every source. A page that fails is reported and skipped, not fatal."""
    pages, failures = [], []
    for source in sources:
        try:
            passages = fetch(source["url"], cache_dir=cache_dir)
        except requests.RequestException as e:
            failures.append((source["name"], str(e)))
            continue
        if passages:
            pages.append({**source, "passages": passages})
        else:
            failures.append((source["name"], "no readable text found"))
    return pages, failures
