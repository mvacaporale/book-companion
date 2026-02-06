"""Base class for book parsers."""

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
        """Extract a title from the file name."""
        return file_path.stem.replace("_", " ").replace("-", " ").title()


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
