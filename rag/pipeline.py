"""The assistant: retrieve, generate, remember the conversation, and evaluate."""
from .chunk import chunk_pages
from .generate import NO_ANSWER
from .index import VectorIndex
from .scrape import fetch_all


class Assistant:
    def __init__(self, index, generator, top_k=4, min_score=0.25):
        self.index = index
        self.generator = generator
        self.top_k = top_k
        self.min_score = min_score
        self.history = []

    @classmethod
    def from_sources(cls, sources, embedder, generator, cache_dir=None, **kwargs):
        pages, failures = fetch_all(sources, cache_dir=cache_dir)
        records = chunk_pages(pages)
        assistant = cls(VectorIndex(embedder, records, include_source=True), generator, **kwargs)
        assistant.pages, assistant.failures = pages, failures
        return assistant

    def retrieve(self, question):
        """Search with the question, plus the previous question for follow-ups."""
        query = question
        if self.history:
            query = f"{self.history[-1][0]} {question}"
        return self.index.search(query, self.top_k)

    def ask(self, question):
        """Answer a question. Returns the answer and the chunks it was based on."""
        chunks = self.retrieve(question)
        if not chunks or chunks[0]["score"] < self.min_score:
            # Nothing in the sources is close to the question: do not ask the model to improvise
            answer, used = NO_ANSWER, []
        else:
            answer, used = self.generator.generate(question, chunks, self.history)
        self.history.append((question, answer))
        return {"question": question, "answer": answer, "sources": used}

    def reset(self):
        self.history = []


def evaluate_retrieval(index, questions, ks=(1, 3, 5)):
    """Share of questions whose expected source appears in the top k results."""
    rows = []
    for question, expected in questions:
        results = index.search(question, max(ks))
        rank = next((i for i, r in enumerate(results, start=1) if r["source"] == expected), None)
        rows.append({"question": question, "expected": expected, "rank": rank,
                     "top_source": results[0]["source"], "top_score": round(results[0]["score"], 3)})
    hit_rates = {f"hit@{k}": sum(r["rank"] is not None and r["rank"] <= k for r in rows) / len(rows) for k in ks}
    return rows, hit_rates
