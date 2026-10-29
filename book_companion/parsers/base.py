"""Base class for book parsers."""

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from book_companion.models import BookFormat, Chapter, ParsedBook


class BookParser(ABC):
    """Abstract base class for book parsers."""

    format: BookFormat

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedBook:
        """Parse a book file and return structured content.

        Args:
            file_path: Path to the book file

        Returns:
            ParsedBook with title, author, chapters, and raw text
        """
        pass

    @abstractmethod
    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the given file.

        Args:
            file_path: Path to the book file

        Returns:
            True if this parser can handle the file
        """
        pass

    def _extract_title_from_path(self, file_path: Path) -> str:
        """Extract a title from the file name.

        Drops download-site tags like "(z-library.sk, 1lib.sk)", a trailing
        "(Author Name)" and "..." truncation marks.
        """
        name = file_path.stem.replace("_", " ")
        name = re.sub(r"\s*\([^)]*\b(?:z-?lib|1lib|libgen|anna'?s archive)[^)]*\)", "", name, flags=re.I)
        name = re.sub(r"\s*\([^()]*\)\s*$", "", name)
        name = re.sub(r"\s*(?:\.\.\.|…)\s*$", "", name)
        name = re.sub(r"(?<=\w)-(?=\w)", " ", name) if " " not in name else name
        name = " ".join(name.split())
        return name.title() if name.islower() else name


# Metadata values PDF/EPUB tools leave behind instead of a real title/author.
_PLACEHOLDER_METADATA = re.compile(
    r"^\[.*\]$|^\(.*\)$|^(?:unknown|untitled|none|null|n/?a|no data|anonymous|author|title"
    r"|microsoft word\b.*|.*\.(?:pdf|docx?|indd|epub|qxd|rtf|txt))$",
    re.IGNORECASE,
)


def clean_metadata_value(value: Optional[str]) -> Optional[str]:
    """Return a metadata title/author, or None when it's a placeholder.

    Catches values like "[No data]", "Unknown", "Microsoft Word - draft.docx"
    and ID-like strings such as "443186283" that have no letters.
    """
    if not value:
        return None
    value = " ".join(value.split())
    if not value or _PLACEHOLDER_METADATA.match(value):
        return None
    if not re.search(r"[^\W\d_]", value):  # no letters at all
        return None
    return value


_NUMBER_WORDS = {
    word: i
    for i, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
    )
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}
_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}

_CHAPTER_LABEL = re.compile(r"^(?:chapter|ch\.?)\s+([0-9]{1,3}|[a-z]+(?:[-\s][a-z]+)?)\b", re.IGNORECASE)
_LEADING_NUMBER = re.compile(r"^(\d{1,3})(?:[.:)]|\s|$)")


def _parse_number(token: str) -> Optional[int]:
    token = token.lower().strip()
    if token.isdigit():
        return int(token)
    if token in _NUMBER_WORDS:
        return _NUMBER_WORDS[token]
    parts = re.split(r"[-\s]", token)
    if parts[0] in _TENS:
        if len(parts) == 1:
            return _TENS[parts[0]]
        if len(parts) == 2 and 0 < _NUMBER_WORDS.get(parts[1], 0) < 10:
            return _TENS[parts[0]] + _NUMBER_WORDS[parts[1]]
    if re.fullmatch(r"[ivxlc]+", token):
        total = 0
        for i, ch in enumerate(token):
            value = _ROMAN[ch]
            nxt = _ROMAN[token[i + 1]] if i + 1 < len(token) else 0
            total += -value if value < nxt else value
        return total or None
    return None


def explicit_chapter_number(chapter: Chapter) -> Optional[int]:
    """The book's own number for a chapter ("CHAPTER 12", "12. Title", "Twelve").

    Looks at the title, then the first few short lines of the text, where EPUBs
    often put "CHAPTER 12" above the chapter's real title.
    """
    title = (chapter.title or "").strip()
    head = [line.strip() for line in chapter.content.splitlines()[:5] if line.strip()]
    for candidate in [title] + [line for line in head[:3] if len(line) <= 40]:
        match = _CHAPTER_LABEL.match(candidate)
        if match:
            number = _parse_number(match.group(1))
            if number is not None:
                return number
    match = _LEADING_NUMBER.match(title)
    if match:
        return int(match.group(1))
    if title and (title.lower() in _NUMBER_WORDS or title.lower() in _TENS):
        return _parse_number(title)
    return None


def number_chapters(chapters: list[Chapter]) -> list[Chapter]:
    """Number chapters the way the book does, so "CHAPTER 12" is chapter 12.

    When at least two sections carry explicit, strictly increasing numbers,
    those keep them; unnumbered sections (introduction, prologue, part openers,
    epilogue) get numbers after the last real chapter, in reading order, so
    every number stays unique. Otherwise sections are numbered 1..N in order.
    """
    found = [explicit_chapter_number(ch) for ch in chapters]
    numbers = [n for n in found if n is not None]
    consistent = (
        len(numbers) >= 2
        and numbers[0] <= 2
        and all(a < b for a, b in zip(numbers, numbers[1:]))
    )

    if not consistent:
        for i, chapter in enumerate(chapters, 1):
            chapter.number = i
        return chapters

    next_extra = max(numbers) + 1
    for chapter, number in zip(chapters, found):
        if number is None:
            number = next_extra
            next_extra += 1
        chapter.number = number
    return chapters


def get_parser(file_path: Path) -> Optional[BookParser]:
    """Get the appropriate parser for a file.

    Args:
        file_path: Path to the book file

    Returns:
        A BookParser instance or None if no parser supports the file
    """
    from .pdf_parser import PDFParser
    from .epub_parser import EPUBParser
    from .markdown_parser import MarkdownParser

    parsers = [PDFParser(), EPUBParser(), MarkdownParser()]

    for parser in parsers:
        if parser.can_parse(file_path):
            return parser

    return None
