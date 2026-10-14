# Vibration monitoring RAG assistant

[![tests](https://github.com/Sammyjoseph999/vibration-monitoring-rag/actions/workflows/tests.yml/badge.svg)](https://github.com/Sammyjoseph999/vibration-monitoring-rag/actions/workflows/tests.yml)

**[Try the live demo](https://huggingface.co/spaces/sammyjoseph/vibration-monitoring-assistant)**: runs in your browser, no sign-up or API key.

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

There are two versions of the same assistant: the same sources, chunks, models, prompt and refusal threshold.

### In the browser (the live demo)

`web/index.html` runs both models in the visitor's browser with [Transformers.js](https://huggingface.co/docs/transformers.js), so it can be hosted as a free static page and nothing typed leaves the browser. The passages are embedded in advance by `web/build_index.py`, which uses the same `rag` package as everything else; questions are embedded in the browser with the same model, and the similarity scores match the Python index. The first visit downloads about 370 MB of model files, which the browser then caches.

Two differences from the Python version: answers use greedy decoding rather than beam search, and the answer model is an 8-bit quantised copy of flan-t5-base.

```bash
python web/build_index.py           # writes web/index.json
python -m http.server -d web 8000   # then open http://localhost:8000
```

To publish it, create a Hugging Face Space with the **Static** SDK and upload `web/index.html` and `web/index.json`.

### With Gradio

`app.py` is a chat interface built with Gradio around the Python assistant. It keeps each visitor's conversation separate, so follow-up questions work.

```bash
pip install -r requirements-app.txt
python app.py
```

It opens at http://127.0.0.1:7860. The first start downloads the two models and the eight pages, which takes a few minutes. Gradio Spaces on Hugging Face's free CPU now need a paid account, which is why the live demo is the static version.

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
