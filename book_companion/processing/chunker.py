"""Semantic text chunking with chapter awareness."""

import re
from typing import Optional

from book_companion.models import Chunk, ChunkMetadata, ParsedBook


class Chunker:
    """Chunk text with chapter awareness and semantic boundaries."""

    def __init__(
        self,
        chunk_size: int = 1500,
        overlap: int = 200,
        min_chunk_size: int = 100,
    ):
        """Initialize the chunker.

        Args:
            chunk_size: Target chunk size in characters (~300-400 tokens)
            overlap: Overlap between chunks in characters
            min_chunk_size: Minimum chunk size to avoid tiny chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size

    def chunk_book(self, book: ParsedBook, book_id: str) -> list[Chunk]:
        """Chunk a parsed book into smaller pieces.

        Chunks respect chapter boundaries - no chunk spans multiple chapters.

        Args:
            book: The parsed book to chunk
            book_id: ID of the book for metadata

        Returns:
            List of Chunk objects with metadata
        """
        chunks = []

        for chapter in book.chapters:
            chapter_chunks = self._chunk_chapter(
                text=chapter.content,
                book_id=book_id,
                chapter_title=chapter.title,
                chapter_number=chapter.number,
                start_page=chapter.start_page,
            )
            chunks.extend(chapter_chunks)

        return chunks

    def _chunk_chapter(
        self,
        text: str,
        book_id: str,
        chapter_title: Optional[str],
        chapter_number: int,
        start_page: Optional[int],
    ) -> list[Chunk]:
        """Chunk a single chapter.

        Uses semantic boundaries (paragraphs, sentences) when possible.
        """
        if len(text) <= self.chunk_size:
            # Chapter fits in one chunk
            return [
                Chunk(
                    text=text.strip(),
                    metadata=ChunkMetadata(
                        book_id=book_id,
                        chapter_title=chapter_title,
                        chapter_number=chapter_number,
                        page_number=start_page,
                        start_char=0,
                        end_char=len(text),
                    ),
                )
            ]

        chunks = []
        paragraphs = self._split_into_paragraphs(text)

        current_chunk = ""
        current_start = 0
        char_offset = 0

        for paragraph in paragraphs:
            paragraph_with_space = paragraph + "\n\n"

            if len(current_chunk) + len(paragraph_with_space) <= self.chunk_size:
                # Add paragraph to current chunk
                current_chunk += paragraph_with_space
            else:
                # Save current chunk if it meets minimum size
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append(
                        Chunk(
                            text=current_chunk.strip(),
                            metadata=ChunkMetadata(
                                book_id=book_id,
                                chapter_title=chapter_title,
                                chapter_number=chapter_number,
                                page_number=start_page,
                                start_char=current_start,
                                end_char=char_offset,
                            ),
                        )
                    )

                # Start new chunk with overlap
                if self.overlap > 0 and current_chunk:
                    # Try to find a sentence boundary for overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + paragraph_with_space
                    current_start = max(0, char_offset - len(overlap_text))
                else:
                    current_chunk = paragraph_with_space
                    current_start = char_offset

                # If single paragraph is too long, split it
                if len(current_chunk) > self.chunk_size:
                    sentence_chunks = self._chunk_long_paragraph(
                        current_chunk,
                        book_id,
                        chapter_title,
                        chapter_number,
                        start_page,
                        current_start,
                    )
                    chunks.extend(sentence_chunks[:-1])  # Add all but last
                    current_chunk = sentence_chunks[-1].text if sentence_chunks else ""
                    current_start = char_offset

            char_offset += len(paragraph_with_space)

        # Don't forget the last chunk
        if current_chunk and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(
                Chunk(
                    text=current_chunk.strip(),
                    metadata=ChunkMetadata(
                        book_id=book_id,
                        chapter_title=chapter_title,
                        chapter_number=chapter_number,
                        page_number=start_page,
                        start_char=current_start,
                        end_char=len(text),
                    ),
                )
            )

        return chunks

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs."""
        # Split on double newlines or more
        paragraphs = re.split(r"\n\s*\n", text)
        # Filter out empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]

    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text, preferring sentence boundaries."""
        if len(text) <= self.overlap:
            return text

        # Get the last `overlap` characters
        overlap_text = text[-self.overlap:]

        # Try to start at a sentence boundary
        sentence_starts = list(re.finditer(r"(?<=[.!?])\s+", overlap_text))
        if sentence_starts:
            # Start from the first sentence in the overlap
            start_pos = sentence_starts[0].end()
            return overlap_text[start_pos:]

        return overlap_text

    def _chunk_long_paragraph(
        self,
        text: str,
        book_id: str,
        chapter_title: Optional[str],
        chapter_number: int,
        start_page: Optional[int],
        base_offset: int,
    ) -> list[Chunk]:
        """Chunk a paragraph that's too long by sentence boundaries."""
        sentences = self._split_into_sentences(text)

        chunks = []
        current_chunk = ""
        current_start = base_offset

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(
                        Chunk(
                            text=current_chunk.strip(),
                            metadata=ChunkMetadata(
                                book_id=book_id,
                                chapter_title=chapter_title,
                                chapter_number=chapter_number,
                                page_number=start_page,
                                start_char=current_start,
                                end_char=current_start + len(current_chunk),
                            ),
                        )
                    )
                    current_start += len(current_chunk)
                current_chunk = sentence + " "

        if current_chunk:
            chunks.append(
                Chunk(
                    text=current_chunk.strip(),
                    metadata=ChunkMetadata(
                        book_id=book_id,
                        chapter_title=chapter_title,
                        chapter_number=chapter_number,
                        page_number=start_page,
                        start_char=current_start,
                        end_char=current_start + len(current_chunk),
                    ),
                )
            )

        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting - handles common cases
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]
