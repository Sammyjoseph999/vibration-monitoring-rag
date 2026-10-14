# Vibration monitoring RAG assistant

[![tests](https://github.com/Sammyjoseph999/vibration-monitoring-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/Sammyjoseph999/vibration-monitoring-rag/actions/workflows/tests.yml)

A retrieval-augmented question-answering assistant for vibration monitoring and condition-monitoring products. It reads a set of web pages (vendor product pages and background articles), indexes them, and answers questions using only what it retrieved, listing the sources behind each answer.

It runs entirely on a laptop with open models, so no API key is required.

This project brings together several earlier Colab experiments on the same idea (scraping vendor sites, chunking, embedding, and answering with different LLMs) into one tested package.

## How it works

```
web pages -> clean text -> sentence chunks -> embeddings -> cosine search -> prompt -> answer + sources
 scrape.py                  chunk.py          index.py                      generate.py   pipeline.py
```

| Stage | What it does |
|---|---|
| Scrape | Downloads each page once, keeps headings, paragraphs and list items, drops scripts, navigation, repeated blocks and formula markup |
| Chunk | Groups whole sentences into chunks of up to 120 words, with one sentence of overlap |
| Index | Embeds chunks with `all-MiniLM-L6-v2` and searches by cosine similarity |
| Generate | Answers with `google/flan-t5-base` locally, trimming the context to fit the model's 512-token input |
| Assistant | Keeps conversation history for follow-up questions and declines when nothing relevant is retrieved |

## Results

On ten test questions, each with a known source:

| Index | hit@1 | hit@3 | hit@5 |
|---|---|---|---|
| Chunk text only | 90% | 90% | 90% |
| Source name + chunk text | 100% | 100% | 100% |

The one miss was a question about Fluke's products. The Fluke page rarely repeats the company name, so its chunks did not say whose products they described. Embedding the source name with each chunk fixed it. Ten hand-written questions are a small test; the notebook discusses the limits.

Off-topic questions (a bread recipe, a football result) score below 0.15 similarity against the index and are declined without calling the generator.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

The first run downloads the two models (about 1.1 GB) and the eight pages. Pages are cached in `data/cache/`, which is not committed because the text belongs to its publishers.

To use a hosted chat model in place of the local one, set one of these before starting, for example in a `.env` file that you load yourself:

```bash
export GROQ_API_KEY="your-key"      # uses llama-3.3-70b-versatile
export OPENAI_API_KEY="your-key"    # uses gpt-4o-mini
```

Keys are read from the environment only. Never put a key in a notebook cell.

Using it from Python:

```python
from rag.generate import load_generator
from rag.index import load_embedder
from rag.pipeline import Assistant
from rag.sources import SOURCES

assistant = Assistant.from_sources(SOURCES, load_embedder(), load_generator(), cache_dir="data/cache")
result = assistant.ask("What is condition monitoring?")
print(result["answer"])
for chunk in result["sources"]:
    print(chunk["source"], chunk["url"])
```

## Web demo

`app.py` is a small chat interface built with Gradio. It answers each question and lists the passages it used, and keeps each visitor's conversation separate so follow-up questions work.

```bash
pip install -r requirements-app.txt
python app.py
```

It opens at http://127.0.0.1:7860. The first start downloads the two models and the eight pages, which takes a few minutes.

### Deploying it as a Hugging Face Space

1. Create a free account at huggingface.co and a new Space with the **Gradio** SDK and the free CPU hardware.
2. Upload `app.py`, the `rag/` folder, and `requirements-app.txt` renamed to `requirements.txt`.
3. The Space builds and starts on its own. Add its link to the top of this README.

## Tests

```bash
pip install -r requirements-test.txt
pytest
```

The 19 tests cover cleaning, chunking, search, prompt construction, generator selection and the assistant's memory and refusal behaviour. They use small stand-in models, so they need no network access and no downloads.

## Limits and next steps

- `flan-t5-base` mostly copies a sentence or two from the context. It is easy to verify but does not compare or summarise across sources.
- The similarity threshold for declining (0.25) was set by inspection.
- Vendor pages change, so answers will drift over time.
- Next: add PDF manuals as a source, build a larger question set with graded answers, and add a reranker for product comparisons.

## Sources

Vendor pages from Fluke, Erbessd Instruments, Acoem and Emerson, and Wikipedia articles on condition monitoring, predictive maintenance, vibration and accelerometers (CC BY-SA). The repository contains no copy of their text; the notebook shows short excerpts with attribution.

## License

The code is released under the [MIT License](LICENSE). Datasets keep the terms of their original sources, described above.
