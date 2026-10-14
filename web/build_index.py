"""Build the search index for the in-browser demo.

Fetches and chunks the source pages exactly as the Python assistant does, embeds each
chunk with all-MiniLM-L6-v2, and writes everything the page needs to web/index.json.
Run from the repository root:  python web/build_index.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.chunk import chunk_pages  # noqa: E402
from rag.generate import INSTRUCTIONS, NO_ANSWER  # noqa: E402
from rag.index import VectorIndex, load_embedder  # noqa: E402
from rag.scrape import fetch_all  # noqa: E402
from rag.sources import SOURCES  # noqa: E402

pages, failures = fetch_all(SOURCES, cache_dir=ROOT / "data" / "cache")
if failures:
    print("Could not fetch:", failures)
records = chunk_pages(pages)
index = VectorIndex(load_embedder(), records, include_source=True)

out = {
    "instructions": INSTRUCTIONS,
    "no_answer": NO_ANSWER,
    "top_k": 4,
    "min_score": 0.25,
    "chunks": [{"text": r["text"], "source": r["source"], "url": r["url"],
                "vector": [round(float(v), 5) for v in vec]}
               for r, vec in zip(records, index.vectors)],
}
path = ROOT / "web" / "index.json"
path.write_text(json.dumps(out, separators=(",", ":")))
print(f"{len(records)} chunks from {len(pages)} pages -> {path.relative_to(ROOT)} "
      f"({path.stat().st_size / 1e6:.1f} MB)")
