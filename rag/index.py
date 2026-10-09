"""Embed chunks and search them by cosine similarity."""
import numpy as np

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedder(model_name=DEFAULT_EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


class VectorIndex:
    """Exact cosine-similarity search over a list of chunk records.

    The corpus here is a few hundred chunks, so one matrix product per query
    is exact and takes well under a millisecond. An approximate index such as
    FAISS only starts to pay off at hundreds of thousands of vectors.
    """

    def __init__(self, embedder, records, include_source=False):
        """With include_source=True, each chunk is embedded as "<source>: <text>".

        Product pages rarely repeat the company name in every paragraph, so a
        chunk on its own often does not say whose product it describes.
        """
        if not records:
            raise ValueError("Cannot build an index from zero chunks.")
        self.embedder = embedder
        self.records = records
        texts = [f"{r['source']}: {r['text']}" if include_source else r["text"] for r in records]
        self.vectors = self._embed(texts)

    def __len__(self):
        return len(self.records)

    @property
    def dimension(self):
        return self.vectors.shape[1]

    def _embed(self, texts):
        vectors = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype="float32")

    def search(self, query, top_k=4):
        """Return the top_k most similar chunks, each with its similarity score."""
        # Inner product of unit vectors is cosine similarity
        scores = self.vectors @ self._embed([query])[0]
        best = np.argsort(-scores)[:top_k]
        return [{**self.records[i], "score": float(scores[i])} for i in best]
