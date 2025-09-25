"""
pdf_utils.py
PDF extraction and sentence-aware chunking utilities.

Functions:
- extract_text_from_pdf_filelike(file_like) -> str
- sentence_chunking_with_overlap(text, chunk_size, overlap) -> List[str]
"""

import io
import re
from typing import List
import PyPDF2


def extract_text_from_pdf_filelike(file_like: io.BytesIO) -> str:
    """
    Extract text from a file-like object containing PDF bytes.
    Uses PyPDF2.PdfReader and concatenates page text.
    Raises RuntimeError on failure.
    """
    try:
        reader = PyPDF2.PdfReader(file_like)
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {e}")


def sentence_chunking_with_overlap(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Produce sentence-aware chunks approximately 'chunk_size' characters each,
    then apply a character-level overlap between successive chunks.
    Returns a list of chunk strings.
    """
    if not text:
        return []

    # Rough-but-practical sentence split
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current = ""
    for sent in sentences:
        # keep adding sentences until chunk_size reached
        if len(current) + len(sent) + 1 <= chunk_size:
            current += sent + " "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sent + " "

    if current.strip():
        chunks.append(current.strip())

    # Apply overlap (prepend trailing overlap from previous chunk)
    if overlap > 0 and len(chunks) > 1:
        overlapped = []
        for i, ch in enumerate(chunks):
            if i == 0:
                overlapped.append(ch)
            else:
                prev = chunks[i - 1]
                tail = prev[-overlap:] if len(prev) >= overlap else prev
                if not ch.startswith(tail):
                    ch = tail + " " + ch
                overlapped.append(ch)
        chunks = overlapped

    return chunks
