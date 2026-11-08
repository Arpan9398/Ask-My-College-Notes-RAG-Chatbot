"""PDF text extraction that keeps page numbers and fails with friendly messages."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass

from pypdf import PdfReader


class PDFLoadError(Exception):
    """Raised with a user-friendly message when a PDF cannot be used."""


@dataclass(frozen=True)
class PageText:
    source: str
    page: int  # 1-based
    text: str


@dataclass(frozen=True)
class LoadedPDF:
    filename: str
    total_pages: int
    pages: list[PageText]  # only pages that contained extractable text

    @property
    def skipped_pages(self) -> int:
        return self.total_pages - len(self.pages)


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated at line ends
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(data: bytes, filename: str) -> LoadedPDF:
    """Extract text per page. Raises PDFLoadError for unusable files."""
    if not data:
        raise PDFLoadError("The file is empty.")
    if b"%PDF-" not in data[:1024]:
        raise PDFLoadError("This does not look like a valid PDF file.")

    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            try:
                unlocked = reader.decrypt("")
            except Exception:
                unlocked = 0
            if not unlocked:
                raise PDFLoadError("The PDF is password-protected. Remove the password and try again.")
        total_pages = len(reader.pages)
    except PDFLoadError:
        raise
    except Exception as exc:  # pypdf can raise many different error types on corrupt files
        raise PDFLoadError(f"The PDF could not be read (it may be corrupted): {exc}") from exc

    if total_pages == 0:
        raise PDFLoadError("The PDF has no pages.")

    pages: list[PageText] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = _clean(page.extract_text() or "")
        except Exception:
            text = ""  # skip pages that fail to parse rather than failing the whole file
        if text:
            pages.append(PageText(source=filename, page=number, text=text))

    if not pages:
        raise PDFLoadError(
            "No extractable text found. The PDF is probably scanned images; "
            "this app does not do OCR. Run OCR on it first and upload the result."
        )
    return LoadedPDF(filename=filename, total_pages=total_pages, pages=pages)
