"""EPUB parser using ebooklib."""

import re
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub

from book_companion.models import BookFormat, Chapter, ParsedBook
from book_companion.processing.quality import is_non_content_title
from .base import BookParser, clean_metadata_value, number_chapters


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
        title = clean_metadata_value(self._get_metadata(book, "title")) or self._extract_title_from_path(file_path)
        author = clean_metadata_value(self._get_metadata(book, "creator"))

        # Extract chapters in reading (spine) order
        chapters = []
        full_text_parts = []

        for item in self._documents_in_reading_order(book):
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

            chapters.append(
                Chapter(
                    number=len(chapters) + 1,
                    title=chapter_title,
                    content=text.strip(),
                )
            )

        full_text = "\n\n".join(full_text_parts)
        number_chapters(chapters)

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

    def _documents_in_reading_order(self, book: epub.EpubBook) -> list:
        """Spine items first (the reader's order), then any unlisted documents.

        Manifest order, which get_items_of_type returns, often puts notes or
        front matter in odd places.
        """
        documents = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        spine = []
        for entry in book.spine:
            item_id = entry[0] if isinstance(entry, tuple) else entry
            item = book.get_item_with_id(item_id)
            if item is not None and item in documents and item not in spine:
                spine.append(item)
        return spine or documents

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

        return is_non_content_title(title)
