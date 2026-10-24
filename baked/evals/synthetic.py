"""
baked.evals.synthetic — LLM-powered eval pair generation

Use this when you have documents but no labeled query-answer pairs.
An LLM reads each document and generates realistic questions whose answers
are in that document, producing EvalPairs ready for run_eval().

Typical use:

    from baked.evals.synthetic import generate_pairs, openai_llm
    from baked.evals import save_pairs

    docs = [
        ("Wealth is not money...", {"title": "How to Make Wealth", "author": "PG"}),
        ("A startup is a company designed to grow fast...", {"title": "Startup = Growth"}),
    ]

    pairs = generate_pairs(docs, llm=openai_llm(), questions_per_doc=5)
    save_pairs(pairs, "evals/pairs.json")

The LLM is only called during generation — not during eval. Save the pairs
and reuse them to compare embedding models or search configurations cheaply.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from .harness import EvalPair

# Any callable that takes a prompt string and returns the LLM's reply.
LLMFn = Callable[[str], str]

_PROMPT = """\
Read the following text and generate {n} specific, self-contained questions whose \
answers are found within it. Each question should test a different aspect of the content. \
Do not ask vague or generic questions.

Return ONLY a JSON array of strings. No explanation, no markdown fences, no extra text.

Text:
{text}"""


def generate_pairs(
    docs: list[tuple[str, dict[str, Any]]],
    llm: LLMFn,
    questions_per_doc: int = 5,
    text_preview_chars: int = 3000,
) -> list[EvalPair]:
    """
    Generate EvalPairs by prompting an LLM to produce questions from each document.

    Args:
        docs: List of (text, metadata) tuples. metadata becomes EvalPair.expected.
        llm: Callable (str) -> str — use openai_llm() or anthropic_llm() as factories.
        questions_per_doc: Number of questions to generate per document.
        text_preview_chars: How many chars of each doc to send to the LLM.

    Returns:
        List of EvalPairs ready to pass to run_eval().
    """
    pairs: list[EvalPair] = []
    for text, metadata in docs:
        prompt = _PROMPT.format(n=questions_per_doc, text=text[:text_preview_chars])
        response = llm(prompt)
        questions = _extract_json_list(response)
        for q in questions:
            if isinstance(q, str) and q.strip():
                pairs.append(EvalPair(query=q.strip(), expected=metadata))
    return pairs


def _extract_json_list(text: str) -> list:
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return []


def openai_llm(model: str = "gpt-4o-mini", api_key: str | None = None) -> LLMFn:
    """LLM factory using OpenAI Chat Completions (install baked-vector[openai])."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install baked-vector[openai]")

    client = OpenAI(api_key=api_key)

    def llm(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content

    return llm


def groq_llm(model: str = "openai/gpt-oss-120b", api_key: str | None = None) -> LLMFn:
    """LLM factory using Groq's OpenAI-compatible API (install baked-vector[openai])."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install baked-vector[openai]")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )

    def llm(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return resp.choices[0].message.content

    return llm


def anthropic_llm(model: str = "claude-haiku-5-5", api_key: str | None = None) -> LLMFn:
    """LLM factory using Anthropic Messages API (install baked-vector[anthropic])."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install baked-vector[anthropic]")

    client = anthropic.Anthropic(api_key=api_key)

    def llm(prompt: str) -> str:
        msg = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    return llm
