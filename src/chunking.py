"""Split page text into overlapping chunks. Chunks never cross page boundaries,
so every chunk (and every citation) maps to exactly one PDF page."""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .pdf_loader import PageText


def pages_to_chunks(pages: list[PageText], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs: list[Document] = []
    for page in pages:
        for index, piece in enumerate(splitter.split_text(page.text)):
            piece = piece.strip()
            if piece:
                docs.append(
                    Document(
                        page_content=piece,
                        metadata={"source": page.source, "page": page.page, "chunk": index},
                    )
                )
    return docs
