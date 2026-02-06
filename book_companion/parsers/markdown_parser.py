"""Markdown parser."""

import re
from pathlib import Path
from typing import Optional

from book_companion.models import BookFormat, Chapter, ParsedBook
from .base import BookParser


class MarkdownParser(BookParser):
    """Parser for Markdown files."""

    format = BookFormat.MARKDOWN

    def can_parse(self, file_path: Path) -> bool:
        """Check if this is a Markdown file."""
        return file_path.suffix.lower() in (".md", ".markdown", ".txt")

    def parse(self, file_path: Path) -> ParsedBook:
        """Parse a Markdown file and extract chapters from headings."""
        content = file_path.read_text(encoding="utf-8")

        # Extract title from first H1 or filename
        title = self._extract_title(content, file_path)

        # Split into chapters by H1 or H2 headings
        chapters = self._split_into_chapters(content)

        # If no chapters found, create one from full content
        if not chapters:
            chapters = [
                Chapter(
                    number=1,
                    title="Full Document",
                    content=content.strip(),
                )
            ]

        return ParsedBook(
            title=title,
            author=None,
            chapters=chapters,
            format=self.format,
            raw_text=content,
        )

    def _extract_title(self, content: str, file_path: Path) -> str:
        """Extract title from content or filename."""
        # Look for first H1
        h1_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if h1_match:
            return h1_match.group(1).strip()

        # Look for title in YAML frontmatter
        frontmatter_match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
        if frontmatter_match:
            for line in frontmatter_match.group(1).split("\n"):
                if line.startswith("title:"):
                    return line.split(":", 1)[1].strip().strip('"\'')

        # Fall back to filename
        return self._extract_title_from_path(file_path)

    def _split_into_chapters(self, content: str) -> list[Chapter]:
        """Split content into chapters based on headings."""
        # Remove YAML frontmatter
        content = re.sub(r"^---\n.*?\n---\n?", "", content, flags=re.DOTALL)

        # Find all H1 or H2 headings
        heading_pattern = r"^(#{1,2})\s+(.+)$"
        matches = list(re.finditer(heading_pattern, content, re.MULTILINE))

        if len(matches) < 2:
            # Not enough headings for chapters
            return []

        chapters = []
        for i, match in enumerate(matches):
            heading_level = len(match.group(1))
            heading_title = match.group(2).strip()

            # Skip if it looks like a non-chapter section
            if self._is_skip_section(heading_title):
                continue

            # Get content between this heading and the next
            start = match.end()
            if i < len(matches) - 1:
                end = matches[i + 1].start()
            else:
                end = len(content)

            chapter_content = content[start:end].strip()

            if chapter_content:  # Only add if there's content
                chapters.append(
                    Chapter(
                        number=len(chapters) + 1,
                        title=heading_title,
                        content=chapter_content,
                    )
                )

        return chapters

    def _is_skip_section(self, title: str) -> bool:
        """Check if this section should be skipped."""
        title_lower = title.lower()
        skip_patterns = [
            "table of contents",
            "toc",
            "acknowledgment",
            "about the author",
        ]
        return any(pattern in title_lower for pattern in skip_patterns)
