"""Compare answer models on the same retrieved passages.

Retrieval runs once per question and every model gets the same chunks, prompt
and refusal threshold, so only the answer model changes.
"""
import time

from .generate import NO_ANSWER


def retrieve_once(index, questions, top_k=4, min_score=0.25):
    """Search once per question. Questions below the threshold get no chunks (declined)."""
    retrieved = []
    for question, expected in questions:
        chunks = index.search(question, top_k)
        if not chunks or chunks[0]["score"] < min_score:
            chunks = []
        retrieved.append({"question": question, "expected": expected, "chunks": chunks})
    return retrieved


def compare_models(generators, retrieved):
    """Answer every retrieved question with every generator.

    generators maps a display name to an object with generate(question, chunks).
    Returns one row per (question, model).
    """
    rows = []
    for item in retrieved:
        for name, generator in generators.items():
            start = time.perf_counter()
            if item["chunks"]:
                answer, used = generator.generate(item["question"], item["chunks"])
            else:
                answer, used = NO_ANSWER, []
            rows.append({
                "question": item["question"],
                "expected_source": item["expected"],
                "model": name,
                "answer": answer,
                "declined": answer.strip() == NO_ANSWER,
                "words": len(answer.split()),
                "chunks_used": len(used),
                "expected_source_used": any(c["source"] == item["expected"] for c in used),
                "seconds": round(time.perf_counter() - start, 2),
            })
    return rows


def summarise(rows):
    """Per model: questions, declined, mean words, mean chunks used and median latency."""
    summary = {}
    for name in dict.fromkeys(r["model"] for r in rows):
        mine = [r for r in rows if r["model"] == name]
        seconds = sorted(r["seconds"] for r in mine)
        summary[name] = {
            "questions": len(mine),
            "declined": sum(r["declined"] for r in mine),
            "mean_words": round(sum(r["words"] for r in mine) / len(mine), 1),
            "mean_chunks_used": round(sum(r["chunks_used"] for r in mine) / len(mine), 1),
            "median_seconds": seconds[len(seconds) // 2],
        }
    return summary
