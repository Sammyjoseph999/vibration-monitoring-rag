"""Compare flan-t5-base with a hosted chat model and a hosted reasoning model.

Needs GROQ_API_KEY or OPENAI_API_KEY, in the environment or in a .env file at the
repository root. Run from the repository root:  python scripts/compare_models.py
Writes results/model_comparison.csv (every answer) and prints a summary.
"""
import csv
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.chunk import chunk_pages  # noqa: E402
from rag.compare import compare_models, retrieve_once, summarise  # noqa: E402
from rag.generate import LocalGenerator, hosted_generators  # noqa: E402
from rag.index import VectorIndex, load_embedder  # noqa: E402
from rag.scrape import fetch_all  # noqa: E402
from rag.sources import EVAL_QUESTIONS, SOURCES  # noqa: E402


def read_env_file(path):
    """KEY=value lines from a .env file. Values are never printed."""
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip().removeprefix("export ").strip()] = value.strip().strip("\"'")
    return values


environ = {**read_env_file(ROOT / ".env"), **os.environ}
hosted = hosted_generators(environ)
if not hosted:
    sys.exit("Set GROQ_API_KEY or OPENAI_API_KEY (environment or .env) to run the comparison.")

pages, failures = fetch_all(SOURCES, cache_dir=ROOT / "data" / "cache")
if failures:
    print("Could not fetch:", failures)
index = VectorIndex(load_embedder(), chunk_pages(pages), include_source=True)
retrieved = retrieve_once(index, EVAL_QUESTIONS)

generators = {"local (flan-t5-base)": LocalGenerator(), **hosted}
rows = compare_models(generators, retrieved)

out = ROOT / "results" / "model_comparison.csv"
out.parent.mkdir(exist_ok=True)
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print("| Model | Questions | Declined | Mean words | Median seconds |")
print("|---|---|---|---|---|")
for name, s in summarise(rows).items():
    print(f"| {name} | {s['questions']} | {s['declined']} | {s['mean_words']} | {s['median_seconds']} |")
print(f"\nEvery answer: {out.relative_to(ROOT)}")
