"""PostgreSQL vector store using pgvector extension."""

from typing import Optional

from book_companion.models import Chunk, ChunkMetadata, RetrievedContext
from .database import get_cursor


class PgVectorStore:
    """PostgreSQL-based vector store for book chunks using pgvector.

    Drop-in replacement for ChromaDB VectorStore with the same interface.
    """

    def __init__(self, persist_dir=None):
        """Initialize the vector store.

        Args:
            persist_dir: Ignored. Kept for API compatibility with ChromaDB VectorStore.
        """
        # persist_dir is ignored - we use database connection from environment
        pass

    def add_chunks(self, book_id: str, chunks: list[Chunk]) -> None:
        """Add chunks to the vector store.

        Args:
            book_id: The book ID
            chunks: List of chunks with embeddings
        """
        if not chunks:
            return

        # Filter to chunks that have embeddings
        valid_chunks = [c for c in chunks if c.embedding]
        if not valid_chunks:
            return

        # Insert in batches to avoid memory issues
        batch_size = 500
        for i in range(0, len(valid_chunks), batch_size):
            batch = valid_chunks[i : i + batch_size]
            self._insert_batch(book_id, batch)

    def _insert_batch(self, book_id: str, chunks: list[Chunk]) -> None:
        """Insert a batch of chunks."""
        with get_cursor() as cur:
            # Use executemany with properly formatted embedding arrays
            values = []
            for chunk in chunks:
                # Format embedding as PostgreSQL vector literal
                embedding_str = "[" + ",".join(str(x) for x in chunk.embedding) + "]"
                values.append(
                    (
                        chunk.id,
                        book_id,
                        chunk.text,
                        embedding_str,
                        chunk.metadata.chapter_title,
                        chunk.metadata.chapter_number,
                        chunk.metadata.page_number,
                        chunk.metadata.start_char,
                        chunk.metadata.end_char,
                    )
                )

            # Batch insert
            cur.executemany(
                """
                INSERT INTO chunks (
                    id, book_id, text, embedding,
                    chapter_title, chapter_number, page_number,
                    start_char, end_char
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding,
                    chapter_title = EXCLUDED.chapter_title,
                    chapter_number = EXCLUDED.chapter_number,
                    page_number = EXCLUDED.page_number,
                    start_char = EXCLUDED.start_char,
                    end_char = EXCLUDED.end_char
                """,
                values,
            )

    def query(
        self,
        book_id: str,
        query_embedding: list[float],
        n_results: int = 8,
    ) -> RetrievedContext:
        """Query the vector store for similar chunks.

        Args:
            book_id: The book ID to search
            query_embedding: The query embedding vector
            n_results: Number of results to return

        Returns:
            RetrievedContext with chunks and formatted context
        """
        # Format embedding as PostgreSQL vector literal
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        with get_cursor(dict_cursor=True) as cur:
            # Use cosine distance operator (<=>)
            cur.execute(
                """
                SELECT
                    id, text, chapter_title, chapter_number,
                    page_number, start_char, end_char,
                    embedding <=> %s::vector AS distance
                FROM chunks
                WHERE book_id = %s
                ORDER BY distance ASC
                LIMIT %s
                """,
                (embedding_str, book_id, n_results),
            )
            rows = cur.fetchall()

        chunks = []
        for row in rows:
            chunk = Chunk(
                id=row["id"],
                text=row["text"],
                metadata=ChunkMetadata(
                    book_id=book_id,
                    chapter_title=row["chapter_title"],
                    chapter_number=row["chapter_number"],
                    page_number=row["page_number"],
                    start_char=row["start_char"],
                    end_char=row["end_char"],
                ),
            )
            chunks.append(chunk)

        # Format context for LLM
        formatted_context = self._format_context(chunks)
        chunk_ids = [c.id for c in chunks]

        return RetrievedContext(
            chunks=chunks,
            formatted_context=formatted_context,
            chunk_ids=chunk_ids,
        )

    def _format_context(self, chunks: list[Chunk]) -> str:
        """Format chunks into context for the LLM.

        Includes citations with chapter and page info.
        """
        if not chunks:
            return "No relevant passages found."

        parts = []
        for i, chunk in enumerate(chunks, 1):
            # Build citation
            citation_parts = []
            if chunk.metadata.chapter_title:
                citation_parts.append(f"Chapter: {chunk.metadata.chapter_title}")
            elif chunk.metadata.chapter_number:
                citation_parts.append(f"Chapter {chunk.metadata.chapter_number}")
            if chunk.metadata.page_number:
                citation_parts.append(f"Page {chunk.metadata.page_number}")

            citation = ", ".join(citation_parts) if citation_parts else f"Passage {i}"

            parts.append(f"[{citation}]\n{chunk.text}")

        return "\n\n---\n\n".join(parts)

    def delete_book(self, book_id: str) -> bool:
        """Delete all data for a book.

        Args:
            book_id: The book ID

        Returns:
            True if deletion was successful
        """
        try:
            with get_cursor() as cur:
                cur.execute("DELETE FROM chunks WHERE book_id = %s", (book_id,))
            return True
        except Exception:
            return False

    def get_chunk_count(self, book_id: str) -> int:
        """Get the number of chunks for a book.

        Args:
            book_id: The book ID

        Returns:
            Number of chunks stored
        """
        try:
            with get_cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM chunks WHERE book_id = %s", (book_id,)
                )
                result = cur.fetchone()
                return result[0] if result else 0
        except Exception:
            return 0

    def collection_exists(self, book_id: str) -> bool:
        """Check if chunks exist for a book.

        Args:
            book_id: The book ID

        Returns:
            True if the book has chunks stored
        """
        return self.get_chunk_count(book_id) > 0

    def get_or_create_collection(self, book_id: str):
        """No-op for API compatibility with ChromaDB.

        In PostgreSQL, we don't need explicit collections.
        Returns a stub object for compatibility.
        """

        class _StubCollection:
            """Stub for ChromaDB collection compatibility."""

            def __init__(self, book_id: str, store: "PgVectorStore"):
                self._book_id = book_id
                self._store = store

            def count(self) -> int:
                return self._store.get_chunk_count(self._book_id)

        return _StubCollection(book_id, self)
