"""Vector store using ChromaDB."""

import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from book_companion.models import Chunk, ChunkMetadata, RetrievedContext


def get_data_dir() -> Path:
    """Get the data directory for bookrc."""
    data_dir = Path(os.getenv("BOOKRC_DB_PATH", Path.home() / ".bookrc"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


class VectorStore:
    """ChromaDB-based vector store for book chunks."""

    def __init__(self, persist_dir: Optional[Path] = None):
        """Initialize the vector store.

        Args:
            persist_dir: Directory for ChromaDB persistence.
                        Defaults to ~/.bookrc/db/
        """
        if persist_dir is None:
            persist_dir = get_data_dir() / "db"

        persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

    def _get_collection_name(self, book_id: str) -> str:
        """Get the collection name for a book."""
        return f"book_{book_id}"

    def get_or_create_collection(self, book_id: str) -> chromadb.Collection:
        """Get or create a collection for a book.

        Args:
            book_id: The book ID

        Returns:
            ChromaDB collection for the book
        """
        return self.client.get_or_create_collection(
            name=self._get_collection_name(book_id),
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, book_id: str, chunks: list[Chunk]) -> None:
        """Add chunks to the vector store.

        Args:
            book_id: The book ID
            chunks: List of chunks with embeddings
        """
        if not chunks:
            return

        collection = self.get_or_create_collection(book_id)

        # Prepare data for ChromaDB
        ids = [chunk.id for chunk in chunks]
        embeddings = [chunk.embedding for chunk in chunks if chunk.embedding]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "book_id": chunk.metadata.book_id,
                "chapter_title": chunk.metadata.chapter_title or "",
                "chapter_number": chunk.metadata.chapter_number or 0,
                "page_number": chunk.metadata.page_number or 0,
                "start_char": chunk.metadata.start_char,
                "end_char": chunk.metadata.end_char,
            }
            for chunk in chunks
        ]

        # Add in batches to avoid memory issues
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
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
        collection = self.get_or_create_collection(book_id)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                metadata_dict = results["metadatas"][0][i]
                chunk = Chunk(
                    id=chunk_id,
                    text=results["documents"][0][i],
                    metadata=ChunkMetadata(
                        book_id=metadata_dict["book_id"],
                        chapter_title=metadata_dict.get("chapter_title") or None,
                        chapter_number=metadata_dict.get("chapter_number") or None,
                        page_number=metadata_dict.get("page_number") or None,
                        start_char=metadata_dict["start_char"],
                        end_char=metadata_dict["end_char"],
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
            collection_name = self._get_collection_name(book_id)
            self.client.delete_collection(collection_name)
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
            collection = self.client.get_collection(
                name=self._get_collection_name(book_id)
            )
            return collection.count()
        except Exception:
            return 0

    def collection_exists(self, book_id: str) -> bool:
        """Check if a collection exists for a book.

        Args:
            book_id: The book ID

        Returns:
            True if the collection exists
        """
        try:
            self.client.get_collection(name=self._get_collection_name(book_id))
            return True
        except Exception:
            return False
