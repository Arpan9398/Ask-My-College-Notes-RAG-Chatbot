import io

import pytest
from reportlab.pdfgen import canvas

from src.pdf_loader import PDFLoadError, load_pdf


def make_pdf(pages):
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for text in pages:
        if text:
            c.drawString(72, 750, text)
        c.showPage()
    c.save()
    return buf.getvalue()


def test_extracts_text_with_page_numbers():
    loaded = load_pdf(make_pdf(["alpha page", "beta page"]), "n.pdf")
    assert loaded.total_pages == 2
    assert [(p.page, p.text) for p in loaded.pages] == [(1, "alpha page"), (2, "beta page")]
    assert loaded.pages[0].source == "n.pdf"


def test_blank_pages_are_skipped_but_counted():
    loaded = load_pdf(make_pdf(["first", "", "third"]), "n.pdf")
    assert [p.page for p in loaded.pages] == [1, 3]
    assert loaded.skipped_pages == 1


@pytest.mark.parametrize("data", [b"", b"just some text, not a pdf", b"%PDF-1.4\ngarbage that is truncated"])
def test_invalid_files_raise_friendly_error(data):
    with pytest.raises(PDFLoadError):
        load_pdf(data, "bad.pdf")


def test_pdf_without_text_raises_ocr_hint():
    with pytest.raises(PDFLoadError, match="OCR"):
        load_pdf(make_pdf(["", ""]), "scan.pdf")
