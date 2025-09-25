"""
embeddings.py

Batched embedding calls to OpenAI. This module does NOT persist embeddings to disk;
it returns embeddings for use by the caller. In the current session-only design,
caller (app.py) will store embeddings in st.session_state.

Functions:
- get_embeddings_for_texts(texts: List[str], model: str, batch_size: int) -> List[List[float]]
"""

import time
from typing import List
import numpy as np
from openai import OpenAI
import os

# Create client using env var OPENAI_API_KEY. You can also integrate dotenv externally.
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())  # this loads .env file values into os.environ

_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def get_embeddings_for_texts(texts: List[str], model: str, batch_size: int) -> List[List[float]]:
    """
    Request embeddings in batches. Returns list of vectors (Python lists).
    Raises RuntimeError on API failure.
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            resp = _client.embeddings.create(input=batch, model=model)
            batch_embs = [item.embedding for item in resp.data]
            all_embeddings.extend(batch_embs)
        except Exception as e:
            raise RuntimeError(f"Embedding API failed for batch starting at {i}: {e}")
        # small sleep to be polite / avoid hitting rate limits
        time.sleep(0.05)
    # Convert any numpy-like sequences to plain Python lists (safe to store in session)
    return [list(map(float, emb)) for emb in all_embeddings]
