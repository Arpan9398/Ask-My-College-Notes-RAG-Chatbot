"""FAISS vector store (via LangChain) + Sentence Transformers embeddings.

Several PDFs can live in one index. A small JSON manifest records which chunk ids
belong to which file, so a file can be replaced or removed without re-embedding
everything. The index is persisted to disk and reloaded on the next start.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from .config import Settings
from .models import RetrievedChunk

_INDEX_FILES = ("index.faiss", "index.pkl")


class KnowledgeBaseError(Exception):
    """Raised with a user-friendly message for embedding / index problems."""


def build_embeddings(model_name: str):
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise KnowledgeBaseError(
            "Missing dependency 'langchain-huggingface'. Run: pip install -r requirements.txt"
        ) from exc
    try:
        # Normalised vectors make L2 distance and cosine similarity interchangeable.
        return HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        raise KnowledgeBaseError(
            f"Could not load the embedding model '{model_name}'. The first run needs an "
            f"internet connection to download it (~90 MB). Details: {exc}"
        ) from exc


class NotesIndex:
    def __init__(self, settings: Settings, index_dir: str | Path | None = None, embeddings: Any = None):
        self.settings = settings
        self.index_dir = Path(index_dir) if index_dir else settings.index_dir
        self.embeddings = embeddings or build_embeddings(settings.embedding_model)
        self.store = None  # langchain FAISS vector store, created lazily
        self.files: dict[str, dict] = {}  # filename -> {sha256, pages, chunks, ids}
        self.load_warning: str | None = None
        self._load()

    # ------------------------------------------------------------------ state
    @property
    def is_empty(self) -> bool:
        return self.store is None or not self.files

    @property
    def num_chunks(self) -> int:
        return sum(info["chunks"] for info in self.files.values())

    def _manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    def _load(self) -> None:
        manifest_path = self._manifest_path()
        faiss_ready = all((self.index_dir / name).exists() for name in _INDEX_FILES)
        if not (manifest_path.exists() and faiss_ready):
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("embedding_model") != self.settings.embedding_model:
                raise ValueError("embedding model changed since the index was built")
            from langchain_community.vectorstores import FAISS

            # index.pkl is a pickle written by this app itself; only load indexes you created.
            self.store = FAISS.load_local(
                str(self.index_dir), self.embeddings, allow_dangerous_deserialization=True
            )
            self.files = manifest["files"]
        except Exception as exc:
            self.store, self.files = None, {}
            self.clear()
            self.load_warning = (
                f"The previously saved index could not be used ({exc}). It was reset - "
                "please upload your PDFs again."
            )

    def _save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.store.save_local(str(self.index_dir))
        tmp = self._manifest_path().with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"embedding_model": self.settings.embedding_model, "files": self.files}, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self._manifest_path())

    # ------------------------------------------------------------- mutations
    def add_document(self, filename: str, sha256: str, docs: list, num_pages: int) -> str:
        """Index chunks for one PDF. Returns 'added', 'replaced' or 'unchanged'."""
        existing = self.files.get(filename)
        if existing and existing["sha256"] == sha256:
            return "unchanged"
        if not docs:
            raise KnowledgeBaseError("No text chunks were produced from this PDF.")

        texts = [d.page_content for d in docs]
        metadatas = [d.metadata for d in docs]
        ids = [uuid.uuid4().hex for _ in docs]
        try:
            if self.store is None:
                from langchain_community.vectorstores import FAISS

                self.store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas, ids=ids)
            else:
                self.store.add_texts(texts, metadatas=metadatas, ids=ids)
        except Exception as exc:
            raise KnowledgeBaseError(f"Embedding / indexing failed: {exc}") from exc

        if existing:  # new version is in; now drop the old vectors
            self.store.delete(existing["ids"])
        self.files[filename] = {"sha256": sha256, "pages": num_pages, "chunks": len(ids), "ids": ids}
        self._save()
        return "replaced" if existing else "added"

    def remove_document(self, filename: str) -> None:
        info = self.files.get(filename)
        if not info:
            return
        if len(self.files) == 1:
            self.clear()
            return
        self.store.delete(info["ids"])
        del self.files[filename]
        self._save()

    def clear(self) -> None:
        self.store = None
        self.files = {}
        for name in (*_INDEX_FILES, "manifest.json"):
            path = self.index_dir / name
            if path.exists():
                path.unlink()

    # ---------------------------------------------------------------- search
    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        if self.is_empty or not query.strip():
            return []
        try:
            pairs = self.store.similarity_search_with_score(query, k=k)
        except Exception as exc:
            raise KnowledgeBaseError(f"Searching the notes failed: {exc}") from exc
        chunks = []
        for doc, squared_l2 in pairs:
            # FAISS returns squared L2 distance; for unit vectors cosine = 1 - d/2.
            cosine = max(-1.0, min(1.0, 1.0 - float(squared_l2) / 2.0))
            chunks.append(
                RetrievedChunk(
                    text=doc.page_content,
                    source=str(doc.metadata.get("source", "unknown")),
                    page=int(doc.metadata.get("page", 0)),
                    score=cosine,
                )
            )
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks
