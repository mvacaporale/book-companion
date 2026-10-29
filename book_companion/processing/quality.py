"""Heuristics for telling real book content from front/back matter and junk.

Used at ingest (to skip non-content sections) and at query time (so books
ingested before these rules existed stop returning index pages, endnotes and
bare part headings as search matches).
"""

import re
from typing import Optional

# Section titles that are never the book's argument. Anchored at the start so
# "Discovering ..." isn't caught by "cover" or "Indexing ..." by "index".
_NON_CONTENT_TITLE = re.compile(
    r"^\s*(?:"
    r"index|general index|subject index|notes|endnotes|end notes|footnotes"
    r"|notes (?:and|&) sources|sources|source notes|bibliography|references"
    r"|works cited|selected bibliography|further reading|suggested reading"
    r"|acknowledge?ments?|copyright|contents|table of contents|toc"
    r"|cover|title page|half title|dedication|epigraph"
    r"|about the authors?|about the publisher|also by\b.*|praise for\b.*"
    r"|by the same author|credits|permissions|colophon|list of (?:figures|tables|illustrations)"
    r")\s*[.:]?\s*$",
    re.IGNORECASE,
)

# Index entries end with page locators: "habit loops, 23, 45–47, 112n"
_PAGE_LOCATOR_LINE = re.compile(r"(?:\d{1,4}(?:[-–]\d{1,4})?n?\.?)\s*$")
# Endnotes start with a note number and cite something.
_NOTE_LINE = re.compile(r"^\s*\d{1,3}[.)]?\s+\S")
_CITATION_MARK = re.compile(
    r"\b(?:ibid|op\. cit|et al|pp?\.\s*\d|vol\.|https?://|doi|press|journal|university)\b"
    r"|\(\d{4}\)|\b(?:19|20)\d{2}\b",
    re.IGNORECASE,
)

MIN_CONTENT_WORDS = 25


def is_non_content_title(title: Optional[str]) -> bool:
    """True for front/back matter section titles (index, notes, copyright...)."""
    return bool(title) and bool(_NON_CONTENT_TITLE.match(title))


def is_low_value_chunk(text: str, chapter_title: Optional[str] = None) -> bool:
    """True when a chunk is an index page, endnotes, or a bare heading."""
    if is_non_content_title(chapter_title):
        return True

    if len(text.split()) < MIN_CONTENT_WORDS:
        return True

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 6:
        locator_lines = sum(1 for line in lines if _PAGE_LOCATOR_LINE.search(line))
        if locator_lines / len(lines) >= 0.5:
            return True

        note_lines = [line for line in lines if _NOTE_LINE.match(line)]
        if len(note_lines) / len(lines) >= 0.4:
            cited = sum(1 for line in note_lines if _CITATION_MARK.search(line))
            if cited / len(note_lines) >= 0.5:
                return True

    return False
