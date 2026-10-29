"""PDF parser using PyMuPDF."""

import re
from pathlib import Path
from typing import Optional

import pymupdf

from book_companion.models import BookFormat, Chapter, ParsedBook
from book_companion.processing.quality import is_non_content_title
from .base import BookParser, clean_metadata_value, number_chapters


class PDFParser(BookParser):
    """Parser for PDF files using PyMuPDF."""

    format = BookFormat.PDF

    def can_parse(self, file_path: Path) -> bool:
        """Check if this is a PDF file."""
        return file_path.suffix.lower() == ".pdf"

    def parse(self, file_path: Path) -> ParsedBook:
        """Parse a PDF file and extract chapters.

        Attempts to detect chapters from the table of contents (TOC).
        Falls back to treating the entire document as one chapter.
        """
        doc = pymupdf.open(str(file_path))

        # Extract metadata
        metadata = doc.metadata
        title = clean_metadata_value(metadata.get("title")) or self._extract_title_from_path(file_path)
        author = clean_metadata_value(metadata.get("author"))

        # Get all text first
        full_text_parts = []
        page_texts: list[tuple[int, str]] = []  # (page_num, text)

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            full_text_parts.append(text)
            page_texts.append((page_num + 1, text))

        full_text = "\n".join(full_text_parts)

        # Try to extract chapters from TOC
        toc = doc.get_toc()
        chapters = self._extract_chapters_from_toc(doc, toc, page_texts)

        # If no chapters found, create a single chapter
        if not chapters:
            chapters = [
                Chapter(
                    number=1,
                    title="Full Document",
                    content=full_text,
                    start_page=1,
                    end_page=len(doc),
                )
            ]

        doc.close()

        return ParsedBook(
            title=title,
            author=author,
            chapters=chapters,
            format=self.format,
            total_pages=len(page_texts),
            raw_text=full_text,
        )

    def _extract_chapters_from_toc(
        self,
        doc: pymupdf.Document,
        toc: list,
        page_texts: list[tuple[int, str]],
    ) -> list[Chapter]:
        """Extract chapters using the PDF table of contents.

        Args:
            doc: PyMuPDF document
            toc: Table of contents as [(level, title, page), ...]
            page_texts: List of (page_num, text) tuples
        """
        if not toc:
            return []

        # Use top-level TOC entries (level 1); fall back to levels 1-2
        max_level = 1
        if not any(level == 1 and self._is_chapter_title(title) for level, title, _ in toc):
            max_level = 2
        entries = [(title, page) for level, title, page in toc if level <= max_level]
        chapter_indices = [i for i, (title, _) in enumerate(entries) if self._is_chapter_title(title)]

        if len(chapter_indices) < 2:
            return []

        chapters = []
        for i in chapter_indices:
            title, start_page = entries[i]
            # A chapter ends where the next TOC entry starts, even a skipped one,
            # so notes and the index don't get glued onto the last chapter.
            if i < len(entries) - 1:
                end_page = max(start_page, entries[i + 1][1] - 1)
            else:
                end_page = len(doc)

            # Extract text for this chapter
            chapter_text = ""
            for page_num, text in page_texts:
                if start_page <= page_num <= end_page:
                    chapter_text += text + "\n"

            chapters.append(
                Chapter(
                    number=len(chapters) + 1,
                    title=title.strip(),
                    content=chapter_text.strip(),
                    start_page=start_page,
                    end_page=end_page,
                )
            )

        return number_chapters(chapters)

    def _is_chapter_title(self, title: str) -> bool:
        """Check if a TOC entry looks like a chapter title."""
        title_lower = title.lower().strip()

        # Common chapter indicators
        chapter_patterns = [
            r"^chapter\s+\d+",
            r"^ch\.\s*\d+",
            r"^part\s+\d+",
            r"^section\s+\d+",
            r"^\d+\.",
            r"^\d+\s+",
        ]

        for pattern in chapter_patterns:
            if re.match(pattern, title_lower):
                return True

        if is_non_content_title(title):
            return False

        # Also accept titles that aren't obviously non-chapters
        skip_patterns = [
            "acknowledgment",
            "dedication",
            "copyright",
            "about the author",
            "index",
            "bibliography",
            "references",
            "appendix",
        ]

        for skip in skip_patterns:
            if skip in title_lower:
                return False

        # Accept if it's a reasonable length title
        return 3 <= len(title) <= 100
