"""Retrieval evaluation: does the right source file + page show up in the top-k?

Answerable questions are scored with Hit@1, Hit@3, Hit@k and MRR.
Unanswerable questions (expected_source = null) are checked separately: we report how
many would be rejected by the relevance threshold, i.e. answered "not found" without
calling the LLM. This evaluates RETRIEVAL only - it does not grade the LLM's wording.
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .baseline import TfidfRetriever
from .config import Settings, get_settings
from .models import RetrievedChunk

SearchFn = Callable[[str, int], list[RetrievedChunk]]


def load_questions(path: str | Path) -> list[dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read evaluation questions from {path}: {exc}") from exc
    questions = data["questions"] if isinstance(data, dict) else data
    for q in questions:
        if "question" not in q:
            raise ValueError("Every evaluation entry needs a 'question' field.")
    return questions


def first_correct_rank(results: list[RetrievedChunk], source: str, pages: list[int]) -> int | None:
    """1-based rank of the first retrieved chunk from `source` on one of `pages`, else None."""
    for rank, chunk in enumerate(results, start=1):
        if chunk.source == source and chunk.page in pages:
            return rank
    return None


def evaluate(questions: list[dict], search_fn: SearchFn, k: int, min_relevance: float) -> dict:
    per_question: list[dict] = []
    for q in questions:
        results = search_fn(q["question"], k)
        answerable = bool(q.get("expected_source"))
        rank = (
            first_correct_rank(results, q["expected_source"], list(q.get("expected_pages", [])))
            if answerable
            else None
        )
        per_question.append(
            {
                "id": q.get("id", ""),
                "question": q["question"],
                "answerable": answerable,
                "expected": f'{q["expected_source"]} p.{",".join(map(str, q.get("expected_pages", [])))}'
                if answerable
                else "(not in notes)",
                "rank": rank,
                "top_score": results[0].score if results else None,
                "retrieved": [{"source": c.source, "page": c.page, "score": round(c.score, 4)} for c in results],
            }
        )

    ans = [p for p in per_question if p["answerable"]]
    unans = [p for p in per_question if not p["answerable"]]

    def hit_at(n: int) -> float | None:
        return sum(1 for p in ans if p["rank"] is not None and p["rank"] <= n) / len(ans) if ans else None

    mrr = sum(1 / p["rank"] for p in ans if p["rank"]) / len(ans) if ans else None
    rejected = [p for p in unans if p["top_score"] is None or p["top_score"] < min_relevance]
    return {
        "answerable": {
            "n": len(ans),
            "hit_at_1": hit_at(1),
            "hit_at_3": hit_at(min(3, k)),
            "hit_at_k": hit_at(k),
            "mrr": mrr,
            "lowest_top_score": min((p["top_score"] for p in ans if p["top_score"] is not None), default=None),
        },
        "unanswerable": {
            "n": len(unans),
            "rejected_by_threshold": len(rejected),
            "rejection_rate": len(rejected) / len(unans) if unans else None,
            "highest_top_score": max((p["top_score"] for p in unans if p["top_score"] is not None), default=None),
        },
        "per_question": per_question,
    }


def run_evaluation(
    notes_dir: str | Path,
    questions_path: str | Path,
    backend: str = "minilm",
    k: int = 4,
    settings: Settings | None = None,
    embeddings: Any = None,
    min_relevance: float | None = None,
) -> dict:
    """Index every PDF in `notes_dir` in a throwaway index and evaluate `questions_path`.

    backend: "minilm" = Sentence Transformers + FAISS (the real pipeline), or
             "tfidf"  = offline lexical baseline (no model download).
    """
    from .chunking import pages_to_chunks
    from .pdf_loader import PDFLoadError, load_pdf

    settings = settings or get_settings()
    threshold = settings.min_relevance if min_relevance is None else min_relevance
    questions = load_questions(questions_path)

    pdfs = sorted(Path(notes_dir).glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found in {notes_dir}")

    warnings: list[str] = []
    chunks_by_file: dict[str, tuple[list, int]] = {}
    for pdf in pdfs:
        try:
            loaded = load_pdf(pdf.read_bytes(), pdf.name)
        except PDFLoadError as exc:
            warnings.append(f"{pdf.name}: {exc}")
            continue
        chunks_by_file[pdf.name] = (
            pages_to_chunks(loaded.pages, settings.chunk_size, settings.chunk_overlap),
            len(loaded.pages),
        )
    all_chunks = [c for chunks, _ in chunks_by_file.values() for c in chunks]
    if not all_chunks:
        raise ValueError("None of the PDFs in the notes folder could be read.")

    if backend == "tfidf":
        label = "TF-IDF baseline (scikit-learn)"
        metrics = evaluate(questions, TfidfRetriever(all_chunks).search, k, threshold)
    elif backend == "minilm":
        from .vector_store import NotesIndex

        label = f"Sentence Transformers + FAISS ({settings.embedding_model})"
        with tempfile.TemporaryDirectory() as tmp:
            index = NotesIndex(settings, index_dir=tmp, embeddings=embeddings)
            for name, (chunks, n_pages) in chunks_by_file.items():
                index.add_document(name, name, chunks, n_pages)
            metrics = evaluate(questions, index.search, k, threshold)
    else:
        raise ValueError("backend must be 'minilm' or 'tfidf'")

    return {
        "backend": backend,
        "backend_label": label,
        "k": k,
        "min_relevance": threshold,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "num_documents": len(chunks_by_file),
        "num_chunks": len(all_chunks),
        "warnings": warnings,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **metrics,
    }
