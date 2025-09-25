"""
config.py
Centralized configuration values for the PDF RAG app.
Edit these values to tune chunking, models, batching, and behavior.
"""

import os

# OpenAI model names
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")

# Chunking parameters
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))       # target chars per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))  # chars of overlap between chunks

# Embedding batching
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 64))

# Retrieval & summary
TOP_K = int(os.getenv("TOP_K", 8))                  # top-k chunks to retrieve for RAG
MAP_SUMMARY_TOKENS = int(os.getenv("MAP_SUMMARY_TOKENS", 120))
FINAL_SUMMARY_TOKENS = int(os.getenv("FINAL_SUMMARY_TOKENS", 500))

# Session behavior
SESSION_ONLY = os.getenv("SESSION_ONLY", "true").lower() in ("1", "true", "yes")
