"""Small shared data types."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by a retriever. `score` is cosine similarity (higher = closer)."""

    text: str
    source: str  # PDF file name
    page: int  # 1-based page number in that PDF
    score: float
