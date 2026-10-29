"""Tests for metadata cleanup and chapter numbering."""

from book_companion.models import Chapter
from book_companion.parsers.base import clean_metadata_value, number_chapters


def _chapters(*specs: tuple[str, str]) -> list[Chapter]:
    return [Chapter(number=0, title=t, content=c) for t, c in specs]


def test_placeholder_metadata_is_dropped() -> None:
    for value in ["[No data]", "443186283", "Unknown", "untitled", "Microsoft Word - draft.docx", "book.pdf", "", None]:
        assert clean_metadata_value(value) is None, value


def test_real_metadata_is_kept() -> None:
    assert clean_metadata_value("  James  Suzman ") == "James Suzman"
    assert clean_metadata_value("In Search of Deeper Learning") == "In Search of Deeper Learning"
    assert clean_metadata_value("1984") is None  # digits-only, treated as an ID
    assert clean_metadata_value("Nineteen Eighty-Four") == "Nineteen Eighty-Four"


def test_front_matter_does_not_shift_chapter_numbers() -> None:
    chapters = number_chapters(_chapters(
        ("Foreword", "..."),
        ("Introduction", "..."),
        ("Part One", "..."),
        ("CHAPTER 1", "The Body Thinks\n..."),
        ("The Second One", "CHAPTER TWO\nMinds Extended\n..."),
        ("3. Minds", "..."),
        ("Chapter XII", "..."),
        ("Epilogue", "..."),
    ))
    by_title = {c.title: c.number for c in chapters}
    assert by_title["CHAPTER 1"] == 1
    assert by_title["The Second One"] == 2
    assert by_title["3. Minds"] == 3
    assert by_title["Chapter XII"] == 12
    # Unnumbered sections go after the last chapter, in reading order, and stay unique
    assert [by_title[t] for t in ["Foreword", "Introduction", "Part One", "Epilogue"]] == [13, 14, 15, 16]


def test_unnumbered_books_are_numbered_in_order() -> None:
    chapters = number_chapters(_chapters(("Fire", "..."), ("Ice", "..."), ("Chapter 7", "...")))
    assert [c.number for c in chapters] == [1, 2, 3]


def test_inconsistent_numbers_fall_back_to_order() -> None:
    chapters = number_chapters(_chapters(("Chapter 2", "..."), ("Chapter 1", "..."), ("Chapter 3", "...")))
    assert [c.number for c in chapters] == [1, 2, 3]


def test_filename_title_drops_download_tags() -> None:
    from pathlib import Path

    from book_companion.parsers.pdf_parser import PDFParser

    title = PDFParser()._extract_title_from_path
    assert title(Path("Grit (Angela Duckworth) (z-library.sk, 1lib.sk, z-lib.sk).epub")) == "Grit"
    assert title(Path("affluence-without-abundance.pdf")) == "Affluence Without Abundance"
    assert title(Path("INSPIRED How to Create... (z-library.sk).epub")) == "INSPIRED How to Create"
