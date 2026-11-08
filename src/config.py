"""Central configuration, read from environment variables (and a local .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_NOTES_DIR = ROOT_DIR / "sample_notes"
EVAL_QUESTIONS_PATH = ROOT_DIR / "evaluation" / "eval_questions.json"
DATA_DIR = ROOT_DIR / "data"

load_dotenv(ROOT_DIR / ".env")


def _get_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    try:
        return max(minimum, int(raw)) if raw else default
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    anthropic_model: str
    openai_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    min_relevance: float
    max_upload_mb: int
    index_dir: Path

    @property
    def llm_label(self) -> str:
        model = self.anthropic_model if self.llm_provider == "anthropic" else self.openai_model
        return f"{self.llm_provider}: {model}"


def get_settings() -> Settings:
    chunk_size = _get_int("CHUNK_SIZE", 800, minimum=100)
    overlap = min(_get_int("CHUNK_OVERLAP", 120, minimum=0), chunk_size // 2)
    return Settings(
        llm_provider=(os.getenv("LLM_PROVIDER") or "anthropic").strip().lower(),
        anthropic_model=os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-5-5",
        openai_model=os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
        embedding_model=os.getenv("EMBEDDING_MODEL") or "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        top_k=_get_int("TOP_K", 4),
        min_relevance=_get_float("MIN_RELEVANCE", 0.20),
        max_upload_mb=_get_int("MAX_UPLOAD_MB", 50),
        index_dir=DATA_DIR / "index",
    )
