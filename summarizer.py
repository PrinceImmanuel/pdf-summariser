"""
summarizer.py

Map-Reduce style summarization helpers. Uses the LLM to:
- summarize individual chunks (map)
- aggregate chunk summaries into final document summary (reduce)

Functions:
- summarize_chunk(chunk_text: str, model: str, approx_max_tokens: int) -> str
- aggregate_summaries(chunk_summaries: List[str], model: str, max_tokens: int) -> str
"""

from typing import List
import os
from openai import OpenAI
import time

_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def summarize_chunk(chunk_text: str, model: str, approx_max_tokens: int = 120) -> str:
    """
    Use the LLM to produce a short, factual summary for a single chunk of text.
    Returns the summary string (or an error message string on failure).
    """
    prompt = (
        "You are a factual summarization assistant. Provide a short summary (2-4 short sentences or bullets) "
        "that captures the main points and any notable facts in the text below.\n\n"
        "Text:\n"
        f"{chunk_text}\n\nSummary:"
    )
    try:
        resp = _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful summarization assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=approx_max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # Return an informative placeholder rather than raising — allows pipeline to continue
        return f"(Chunk summarization failed: {e})"


def aggregate_summaries(chunk_summaries: List[str], model: str, max_tokens: int = 500) -> str:
    """
    Combine multiple chunk-level summaries into a single final summary via the LLM.
    """
    combined = "\n\n".join(chunk_summaries)
    prompt = (
        "You are a professional summarization assistant. Combine the following chunk summaries into a single coherent, "
        "concise final summary that highlights the main themes, most important findings, and notable details. Use short paragraphs or bullets.\n\n"
        "Chunk summaries:\n\n"
        f"{combined}\n\nFinal combined summary:"
    )
    try:
        resp = _client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a factual summarization assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(Aggregation failed: {e})"
