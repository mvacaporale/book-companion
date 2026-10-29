"""PDF parser tests against small generated PDFs."""

from pathlib import Path

import pymupdf

from book_companion.parsers.pdf_parser import PDFParser


def _make_pdf(path: Path, toc: list, pages: int, metadata: dict) -> Path:
    doc = pymupdf.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"Page {i + 1} text")
    doc.set_toc(toc)
    doc.set_metadata(metadata)
    doc.save(str(path))
    doc.close()
    return path


def test_pdf_numbers_and_back_matter(tmp_path: Path) -> None:
    pdf = _make_pdf(
        tmp_path / "affluence-without-abundance.pdf",
        toc=[
            [1, "Introduction", 1],
            [1, "Chapter 1 The Old Ways", 2],
            [1, "Chapter 2 Fire", 3],
            [1, "Notes", 4],
            [1, "Index", 5],
        ],
        pages=5,
        metadata={"title": "[No data]", "author": "443186283"},
    )
    book = PDFParser().parse(pdf)

    assert book.title == "Affluence Without Abundance"
    assert book.author is None
    numbers = {c.title: c.number for c in book.chapters}
    assert numbers["Chapter 1 The Old Ways"] == 1
    assert numbers["Chapter 2 Fire"] == 2
    assert "Notes" not in numbers and "Index" not in numbers
    # The last chapter stops before the notes instead of running to the end
    last = next(c for c in book.chapters if c.number == 2)
    assert last.end_page == 3
