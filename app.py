"""Web demo of the assistant, built with Gradio.

Run locally with `python app.py`, or deploy as a Hugging Face Space (see the README).
"""
import gradio as gr

from rag.generate import load_generator
from rag.index import load_embedder
from rag.pipeline import Assistant
from rag.sources import SOURCES

EXAMPLES = [
    "What is condition monitoring?",
    "What is the Machine Defender app used for?",
    "What does the AMS 2140 do?",
    "How does predictive maintenance differ from preventive maintenance?",
    "What causes resonance?",
]

print("Fetching sources and building the index...")
assistant = Assistant.from_sources(SOURCES, load_embedder(), load_generator(), cache_dir="data/cache")
print(f"Ready: {len(assistant.index)} chunks from {len(assistant.pages)} pages")


def format_sources(chunks):
    if not chunks:
        return "_No source was close enough to answer from._"
    lines = ["**Sources**"]
    for chunk in chunks:
        excerpt = chunk["text"][:160].rsplit(" ", 1)[0] + " ..."
        lines.append(f"- [{chunk['source']}]({chunk['url']}) (similarity {chunk['score']:.2f}): {excerpt}")
    return "\n".join(lines)


def respond(message, history):
    """Answer one question, using this visitor's own conversation for follow-ups.

    The assistant object is shared, so its memory is replaced with the
    visitor's history on every call instead of accumulating across visitors.
    """
    turns = [(history[i]["content"], history[i + 1]["content"])
             for i in range(0, len(history) - 1, 2)
             if history[i]["role"] == "user" and history[i + 1]["role"] == "assistant"]
    assistant.history = [(question, answer.split("\n\n**Sources**")[0]) for question, answer in turns]
    result = assistant.ask(message)
    return f"{result['answer']}\n\n{format_sources(result['sources'])}"


demo = gr.ChatInterface(
    respond,
    title="Vibration monitoring assistant",
    description=("Answers questions about vibration monitoring and condition-monitoring products, using only "
                 "four vendor pages and four Wikipedia articles, and shows the passages behind each answer. "
                 "Runs on small open models, so answers are short and mostly quoted from the sources."),
    examples=EXAMPLES,
)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
