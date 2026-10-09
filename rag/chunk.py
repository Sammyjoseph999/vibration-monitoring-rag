"""Split page text into overlapping chunks of whole sentences."""
import re

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text):
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


def chunk_passages(passages, max_words=120, overlap_sentences=1):
    """Group sentences into chunks of at most max_words words.

    Consecutive chunks share overlap_sentences sentences, so a fact that
    straddles a boundary is still retrievable. A single sentence longer than
    max_words becomes its own chunk rather than being cut mid-sentence.
    """
    sentences = [s for passage in passages for s in split_sentences(passage)]
    chunks, current, current_words = [], [], 0

    for sentence in sentences:
        n_words = len(sentence.split())
        if current and current_words + n_words > max_words:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_words = sum(len(s.split()) for s in current)
        current.append(sentence)
        current_words += n_words

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(pages, **kwargs):
    """Chunk every page and keep track of where each chunk came from."""
    records = []
    for page in pages:
        for position, text in enumerate(chunk_passages(page["passages"], **kwargs)):
            records.append({"text": text, "source": page["name"], "url": page["url"], "position": position})
    return records
