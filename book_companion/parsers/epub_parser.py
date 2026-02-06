"""EPUB parser using ebooklib."""

import re
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub

from book_companion.models import BookFormat, Chapter, ParsedBook
from .base import BookParser


class EPUBParser(BookParser):
    """Parser for EPUB files using ebooklib."""

    format = BookFormat.EPUB

    def can_parse(self, file_path: Path) -> bool:
        """Check if this is an EPUB file."""
        return file_path.suffix.lower() == ".epub"

    def parse(self, file_path: Path) -> ParsedBook:
        """Parse an EPUB file and extract chapters."""
        book = epub.read_epub(str(file_path), options={"ignore_ncx": True})

        # Extract metadata
        title = self._get_metadata(book, "title") or self._extract_title_from_path(file_path)
        author = self._get_metadata(book, "creator")

        # Extract chapters from spine
        chapters = []
        full_text_parts = []
        chapter_num = 0

        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            content = item.get_content()
            soup = BeautifulSoup(content, "lxml")

            # Extract text
            text = self._extract_text(soup)
            if not text.strip():
                continue

            full_text_parts.append(text)

            # Try to find chapter title
            chapter_title = self._extract_chapter_title(soup, item.get_name())

            # Skip non-content sections
            if self._is_skip_section(chapter_title):
                continue

            chapter_num += 1
            chapters.append(
                Chapter(
                    number=chapter_num,
                    title=chapter_title,
                    content=text.strip(),
                )
            )

        full_text = "\n\n".join(full_text_parts)

        # If no chapters found, create one from full text
        if not chapters:
            chapters = [
                Chapter(
                    number=1,
                    title="Full Document",
                    content=full_text,
                )
            ]

        return ParsedBook(
            title=title,
            author=author,
            chapters=chapters,
            format=self.format,
            raw_text=full_text,
        )

    def _get_metadata(self, book: epub.EpubBook, field: str) -> Optional[str]:
        """Extract a metadata field from the EPUB."""
        try:
            values = book.get_metadata("DC", field)
            if values:
                return values[0][0]
        except Exception:
            pass
        return None

    def _extract_text(self, soup: BeautifulSoup) -> str:
        """Extract clean text from HTML soup."""
        # Remove script and style elements
        for element in soup(["script", "style", "nav"]):
            element.decompose()

        # Get text
        text = soup.get_text(separator=" ")

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)

        return text

    def _extract_chapter_title(self, soup: BeautifulSoup, filename: str) -> Optional[str]:
        """Extract a chapter title from the HTML."""
        # Try various heading tags
        for tag in ["h1", "h2", "h3", "title"]:
            heading = soup.find(tag)
            if heading:
                title = heading.get_text().strip()
                if title and len(title) < 200:
                    return title

        # Fall back to filename
        name = Path(filename).stem
        # Clean up common patterns
        name = re.sub(r"^(chapter|ch|part|section)[\-_]?", "", name, flags=re.I)
        name = name.replace("_", " ").replace("-", " ").strip()

        return name.title() if name else None

    def _is_skip_section(self, title: Optional[str]) -> bool:
        """Check if this section should be skipped."""
        if not title:
            return False

        title_lower = title.lower()
        skip_patterns = [
            "copyright",
            "table of contents",
            "toc",
            "cover",
            "title page",
            "about the author",
            "acknowledgment",
            "dedication",
        ]

        return any(pattern in title_lower for pattern in skip_patterns)
