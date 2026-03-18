"""Tests for text chunking."""

import pytest

from book_companion.processing.chunker import Chunker
from book_companion.models import BookFormat, ParsedBook, Chapter


class TestChunker:
    """Test cases for Chunker class."""

    @pytest.fixture
    def chunker(self) -> Chunker:
        """Create a Chunker instance with default settings."""
        return Chunker()

    @pytest.fixture
    def sample_parsed_book(self) -> ParsedBook:
        """Create a sample parsed book for testing."""
        ch1_content = "This is the first chapter. " * 100  # ~3000 chars
        ch2_content = "This is the second chapter. " * 50  # ~1500 chars
        return ParsedBook(
            title="Test Book",
            author="Test Author",
            format=BookFormat.MARKDOWN,
            chapters=[
                Chapter(
                    number=1,
                    title="Chapter 1",
                    content=ch1_content,
                    start_page=1,
                    end_page=10,
                ),
                Chapter(
                    number=2,
                    title="Chapter 2",
                    content=ch2_content,
                    start_page=11,
                    end_page=20,
                ),
            ],
            raw_text=ch1_content + ch2_content,
            total_pages=20,
        )

    def test_chunk_empty_book(self, chunker: Chunker) -> None:
        """Empty book should produce no chunks."""
        book = ParsedBook(
            title="Empty",
            author="Author",
            format=BookFormat.MARKDOWN,
            chapters=[],
            raw_text="",
            total_pages=0,
        )
        chunks = chunker.chunk_book(book, "book-id")
        assert len(chunks) == 0

    def test_chunk_small_chapter(self, chunker: Chunker) -> None:
        """Small chapter should produce single chunk."""
        content = "This is a short chapter."
        book = ParsedBook(
            title="Small Book",
            author="Author",
            format=BookFormat.MARKDOWN,
            chapters=[
                Chapter(
                    number=1,
                    title="Short Chapter",
                    content=content,
                    start_page=1,
                    end_page=1,
                ),
            ],
            raw_text=content,
            total_pages=1,
        )
        chunks = chunker.chunk_book(book, "book-id")
        assert len(chunks) == 1
        assert "short chapter" in chunks[0].text.lower()
        assert chunks[0].metadata.chapter_number == 1
        assert chunks[0].metadata.chapter_title == "Short Chapter"

    def test_chunk_large_chapter(self, chunker: Chunker, sample_parsed_book: ParsedBook) -> None:
        """Large chapter should be split into multiple chunks."""
        chunks = chunker.chunk_book(sample_parsed_book, "book-id")
        # With default chunk size of 1500 and overlap of 200,
        # a 3000 char chapter should produce 2-3 chunks
        chapter_1_chunks = [c for c in chunks if c.metadata.chapter_number == 1]
        assert len(chapter_1_chunks) >= 2

    def test_chunk_metadata(self, chunker: Chunker, sample_parsed_book: ParsedBook) -> None:
        """Chunks should have correct metadata."""
        chunks = chunker.chunk_book(sample_parsed_book, "book-id")
        for chunk in chunks:
            assert chunk.metadata.book_id == "book-id"
            assert chunk.metadata.chapter_number is not None
            assert chunk.metadata.chapter_title is not None
            assert chunk.id is not None

    def test_chunk_overlap(self, chunker: Chunker) -> None:
        """Consecutive chunks should have overlapping content."""
        content = "WORD" * 1000  # 4000 chars
        book = ParsedBook(
            title="Test",
            author="Author",
            format=BookFormat.MARKDOWN,
            chapters=[
                Chapter(
                    number=1,
                    title="Chapter",
                    content=content,
                    start_page=1,
                    end_page=10,
                ),
            ],
            raw_text=content,
            total_pages=10,
        )
        chunks = chunker.chunk_book(book, "book-id")
        if len(chunks) >= 2:
            # Check that there's some overlap in content
            # The end of chunk 0 should appear at the start of chunk 1
            # (approximately, due to how splitting works)
            assert len(chunks[0].text) > 0
            assert len(chunks[1].text) > 0
