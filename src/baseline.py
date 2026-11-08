"""Offline lexical baseline (TF-IDF). Same search interface as NotesIndex.

Used by the evaluation script when you want to test the pipeline without
downloading an embedding model, and as a point of comparison for the neural
retriever. It is NOT used by the chat app.
"""
from __future__ import annotations

from typing import Any, Sequence

from .models import RetrievedChunk


class TfidfRetriever:
    def __init__(self, docs: Sequence[Any]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.docs = list(docs)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([d.page_content for d in self.docs])

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        import numpy as np

        if not query.strip() or not self.docs:
            return []
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        order = np.argsort(-scores)[:k]
        return [
            RetrievedChunk(
                text=self.docs[i].page_content,
                source=str(self.docs[i].metadata["source"]),
                page=int(self.docs[i].metadata["page"]),
                score=float(scores[i]),
            )
            for i in order
        ]
