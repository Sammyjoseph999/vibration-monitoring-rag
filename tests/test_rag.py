"""Tests that need no network access and no model downloads."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.chunk import chunk_pages, chunk_passages, split_sentences  # noqa: E402
from rag.generate import NO_ANSWER, build_prompt, load_generator, trim_to_sentence  # noqa: E402
from rag.index import VectorIndex  # noqa: E402
from rag.pipeline import Assistant, evaluate_retrieval  # noqa: E402
from rag.scrape import clean_text, html_to_passages  # noqa: E402


class WordCountEmbedder:
    """Stand-in embedder: one dimension per vocabulary word."""
    VOCAB = ["vibration", "bearing", "sensor", "pump", "price", "goose", "analyzer", "wireless"]

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        vectors = np.array([[text.lower().count(word) for word in self.VOCAB] for text in texts], dtype="float32")
        vectors += 1e-3
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class EchoGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, question, chunks, history=()):
        self.calls.append((question, list(history)))
        return f"answer from {chunks[0]['source']}", chunks


RECORDS = [
    {"text": "The analyzer measures vibration on a bearing.", "source": "A", "url": "a", "position": 0},
    {"text": "A wireless sensor is mounted on the pump.", "source": "B", "url": "b", "position": 0},
    {"text": "The price list is available on request.", "source": "C", "url": "c", "position": 0},
]


# --- scraping ---------------------------------------------------------------

def test_clean_text_keeps_punctuation_and_drops_citations():
    assert clean_text("Vibration   is measured.[3]  It matters!") == "Vibration is measured. It matters!"


def test_html_to_passages_drops_formula_markup():
    html = '<p>The amplitude grows near resonance <span class="mwe-math-element">{\\displaystyle r\\approx 1}</span> as shown here.</p>'
    assert html_to_passages(html) == ["The amplitude grows near resonance as shown here."]


def test_html_to_passages_drops_navigation_scripts_and_repeats():
    html = """<html><body>
      <nav><p>Home Products Support Contact us today</p></nav>
      <script>var tracking = "this should never appear in the text";</script>
      <h2>Vibration sensors</h2>
      <p>Wireless sensors report bearing condition every few minutes.</p>
      <p>Wireless sensors report bearing condition every few minutes.</p>
      <li>Buy now</li>
      <footer><p>Copyright and legal notices for the whole site</p></footer>
    </body></html>"""
    assert html_to_passages(html) == [
        "Vibration sensors.",
        "Wireless sensors report bearing condition every few minutes.",
    ]


# --- chunking ---------------------------------------------------------------

def test_split_sentences():
    assert split_sentences("First one. Second one! Third? Yes.") == ["First one.", "Second one!", "Third?", "Yes."]


def test_split_sentences_does_not_break_decimals():
    assert split_sentences("The range is 0.5 to 2.5 kHz. That is wide.") == ["The range is 0.5 to 2.5 kHz.", "That is wide."]


def test_chunks_respect_the_word_limit_and_overlap():
    passages = [" ".join(f"Sentence number {i} has exactly seven words." for i in range(10))]
    chunks = chunk_passages(passages, max_words=20, overlap_sentences=1)
    assert len(chunks) > 1
    assert all(len(chunk.split()) <= 20 for chunk in chunks)
    for earlier, later in zip(chunks, chunks[1:]):
        assert split_sentences(earlier)[-1] == split_sentences(later)[0]


def test_no_text_is_lost_when_chunking():
    passages = [" ".join(f"Fact {i} is here." for i in range(30))]
    joined = " ".join(chunk_passages(passages, max_words=15, overlap_sentences=0))
    assert joined == passages[0]


def test_a_long_sentence_is_kept_whole():
    long_sentence = " ".join(["word"] * 50) + "."
    assert chunk_passages([long_sentence], max_words=20) == [long_sentence]


def test_chunk_pages_records_the_source():
    pages = [{"name": "A", "url": "http://a", "passages": ["One sentence here."]}]
    assert chunk_pages(pages) == [{"text": "One sentence here.", "source": "A", "url": "http://a", "position": 0}]


# --- index ------------------------------------------------------------------

def test_search_ranks_the_matching_chunk_first():
    index = VectorIndex(WordCountEmbedder(), RECORDS)
    results = index.search("wireless sensor for a pump", top_k=2)
    assert results[0]["source"] == "B"
    assert results[0]["score"] > results[1]["score"]


def test_top_k_larger_than_the_index_is_safe():
    assert len(VectorIndex(WordCountEmbedder(), RECORDS).search("vibration", top_k=10)) == 3


def test_including_the_source_name_makes_it_searchable():
    class NameEmbedder(WordCountEmbedder):
        VOCAB = ["acme", "vibration"]
    records = [{"text": "Our analyzer measures vibration.", "source": "Acme", "url": "a", "position": 0},
               {"text": "This analyzer also measures vibration.", "source": "Other", "url": "b", "position": 0}]
    plain = VectorIndex(NameEmbedder(), records).search("acme vibration analyzer", top_k=2)
    named = VectorIndex(NameEmbedder(), records, include_source=True).search("acme vibration analyzer", top_k=2)
    assert abs(plain[0]["score"] - plain[1]["score"]) < 1e-6
    assert named[0]["source"] == "Acme" and named[0]["score"] > named[1]["score"]


def test_trim_to_sentence():
    assert trim_to_sentence("First sentence. Second one is cut of") == "First sentence."
    assert trim_to_sentence("Complete answer.") == "Complete answer."
    assert trim_to_sentence("no sentence end at all") == "no sentence end at all"


def test_empty_index_is_rejected():
    with pytest.raises(ValueError):
        VectorIndex(WordCountEmbedder(), [])


# --- prompt and generator selection ----------------------------------------

def test_prompt_contains_sources_history_and_question():
    prompt = build_prompt("How much is it?", RECORDS[:1], history=[("What does it measure?", "Vibration.")])
    assert "(A) The analyzer measures vibration" in prompt
    assert "User: What does it measure?" in prompt
    assert prompt.rstrip().endswith("Question: How much is it?\nAnswer:")


def test_hosted_generators_are_chosen_from_the_environment(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "placeholder")
    generator = load_generator()
    assert generator.name == "chat" and "groq" in str(generator.client.base_url)
    monkeypatch.delenv("GROQ_API_KEY")
    monkeypatch.setenv("OPENAI_API_KEY", "placeholder")
    assert load_generator().model_name == "gpt-4o-mini"


# --- assistant --------------------------------------------------------------

def test_assistant_answers_and_remembers():
    assistant = Assistant(VectorIndex(WordCountEmbedder(), RECORDS), EchoGenerator(), top_k=2)
    first = assistant.ask("Which analyzer measures vibration?")
    assert first["answer"] == "answer from A"
    assistant.ask("Is it wireless?")
    assert assistant.generator.calls[1][1] == [("Which analyzer measures vibration?", "answer from A")]
    assistant.reset()
    assert assistant.history == []


def test_assistant_declines_when_nothing_is_relevant():
    generator = EchoGenerator()
    assistant = Assistant(VectorIndex(WordCountEmbedder(), RECORDS), generator, min_score=0.9)
    result = assistant.ask("How do I cook a goose?")
    assert result["answer"] == NO_ANSWER and result["sources"] == []
    assert generator.calls == []


def test_evaluate_retrieval_reports_hit_rates():
    index = VectorIndex(WordCountEmbedder(), RECORDS)
    rows, hits = evaluate_retrieval(index, [("vibration on a bearing", "A"), ("pump sensor", "C")], ks=(1, 3))
    assert rows[0]["rank"] == 1
    assert hits["hit@1"] == 0.5 and hits["hit@3"] == 1.0
