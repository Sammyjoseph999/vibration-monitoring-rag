"""Turn retrieved chunks into an answer.

The default generator is a small open model that runs locally, so the project
works without any API key. Setting GROQ_API_KEY or OPENAI_API_KEY in the
environment switches to a hosted chat model, which gives better answers.
Keys are only ever read from the environment.
"""
import os
import re

DEFAULT_LOCAL_MODEL = "google/flan-t5-base"
NO_ANSWER = "I don't know based on the sources I have."

INSTRUCTIONS = (
    "Answer the question in one or two sentences, using only the context. "
    f'If the context does not contain the answer, reply: "{NO_ANSWER}"'
)


def format_history(history, max_turns=3):
    """Render the last few exchanges so follow-up questions can be understood."""
    lines = []
    for question, answer in history[-max_turns:]:
        lines.append(f"User: {question}")
        lines.append(f"Assistant: {answer}")
    return "\n".join(lines)


def build_prompt(question, chunks, history=()):
    context = "\n\n".join(f"[{i}] ({chunk['source']}) {chunk['text']}" for i, chunk in enumerate(chunks, start=1))
    parts = [INSTRUCTIONS]
    if history:
        parts.append("Earlier conversation:\n" + format_history(list(history)))
    parts.append("Context:\n" + context)
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


def trim_to_sentence(text):
    """Cut an answer that ran out of tokens back to its last complete sentence."""
    end = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
    if text.endswith((".", "!", "?")) or end == -1:
        return text
    return text[:end + 1]


THINK_BLOCK = re.compile(r"<think>.*?(</think>|$)", flags=re.DOTALL | re.IGNORECASE)


def strip_think(text):
    """Remove a reasoning model's <think> section, keeping only the final answer.

    An unclosed <think> (the model ran out of tokens while reasoning) removes
    everything after it, so a half-finished thought is never shown as an answer.
    """
    return THINK_BLOCK.sub("", text).strip()


class LocalGenerator:
    """A sequence-to-sequence model from Hugging Face, run on this machine."""

    name = "local"

    def __init__(self, model_name=DEFAULT_LOCAL_MODEL, max_input_tokens=512, max_new_tokens=64):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

    def fit_chunks(self, question, chunks, history=()):
        """Drop the lowest-ranked chunks until the prompt fits the model's input size.

        Without this the tokenizer silently truncates the end of the prompt,
        which is where the question is.
        """
        kept = list(chunks)
        while len(kept) > 1 and len(self.tokenizer(build_prompt(question, kept, history))["input_ids"]) > self.max_input_tokens:
            kept.pop()
        return kept

    def generate(self, question, chunks, history=()):
        kept = self.fit_chunks(question, chunks, history)
        inputs = self.tokenizer(build_prompt(question, kept, history), return_tensors="pt",
                                truncation=True, max_length=self.max_input_tokens)
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, num_beams=4)
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        if len(output[0]) >= self.max_new_tokens - 2:
            # The model hit its length limit, so the last sentence is probably cut off
            answer = trim_to_sentence(answer)
        return answer, kept


class ChatGenerator:
    """A hosted chat model reached through the OpenAI-compatible API.

    Set reasoning=True for reasoning models: they get a larger token budget for
    their hidden reasoning, no temperature setting (OpenAI's reasoning models
    reject it), and any <think> section is removed from the answer.
    """

    name = "chat"

    def __init__(self, api_key, model_name, base_url=None, max_tokens=300, reasoning=False):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.reasoning = reasoning
        self.max_tokens = 4000 if reasoning and max_tokens == 300 else max_tokens

    def request(self, question, chunks, history=()):
        """The keyword arguments sent to the chat completions endpoint."""
        kwargs = {"model": self.model_name,
                  "messages": [{"role": "user", "content": build_prompt(question, chunks, history)}]}
        if self.reasoning:
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs.update(max_tokens=self.max_tokens, temperature=0)
        return kwargs

    def generate(self, question, chunks, history=()):
        response = self.client.chat.completions.create(**self.request(question, chunks, history))
        return strip_think(response.choices[0].message.content or ""), list(chunks)


# Hosted models for each provider: (chat model, reasoning model)
HOSTED_MODELS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key": "GROQ_API_KEY",
             "chat": "llama-3.3-70b-versatile", "reasoning": "qwen/qwen3-32b"},
    "openai": {"base_url": None, "key": "OPENAI_API_KEY",
               "chat": "gpt-4o-mini", "reasoning": "o4-mini"},
}


def hosted_generators(environ=None):
    """The hosted chat and reasoning generators for the first provider with a key set.

    Returns an empty dict when no key is set.
    """
    environ = os.environ if environ is None else environ
    for provider, config in HOSTED_MODELS.items():
        key = environ.get(config["key"])
        if key:
            return {
                f"{provider} chat ({config['chat']})": ChatGenerator(key, config["chat"], config["base_url"]),
                f"{provider} reasoning ({config['reasoning']})":
                    ChatGenerator(key, config["reasoning"], config["base_url"], reasoning=True),
            }
    return {}


def load_generator():
    """Pick a generator from the environment: Groq, then OpenAI, then the local model."""
    if os.environ.get("GROQ_API_KEY"):
        return ChatGenerator(os.environ["GROQ_API_KEY"], os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                             base_url="https://api.groq.com/openai/v1")
    if os.environ.get("OPENAI_API_KEY"):
        return ChatGenerator(os.environ["OPENAI_API_KEY"], os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    return LocalGenerator()
