"""PostgreSQL storage for book registry, sessions, and book indices."""

import json
from datetime import datetime
from typing import Optional

from book_companion.models import (
    Book,
    BookFormat,
    BookIndex,
    BookRegistry,
    BookSummary,
    ChatMessage,
    ChatRole,
    ChapterIndexEntry,
    ChapterSummary,
    Narrative,
    NarrativeType,
    Session,
)
from .database import get_cursor


class PgBookRegistryStore:
    """PostgreSQL storage for the book registry.

    Drop-in replacement for file-based BookRegistryStore.
    """

    def __init__(self, data_dir=None):
        """Initialize the book registry store.

        Args:
            data_dir: Ignored. Kept for API compatibility.
        """
        # data_dir is ignored - we use database connection from environment
        self._registry: Optional[BookRegistry] = None

    def load(self) -> BookRegistry:
        """Load the registry from database.

        Note: Always queries the database to ensure fresh data.
        This is important because other processes (like ingestion)
        may have modified the database.
        """
        # Always reload from database - don't use cached _registry
        # This ensures we see changes made by other store instances
        self._registry = BookRegistry()

        with get_cursor(dict_cursor=True) as cur:
            cur.execute(
                """
                SELECT id, title, author, format, file_path, file_hash,
                       total_chunks, total_pages, ingested_at,
                       embedding_tokens, summarization_input_tokens,
                       summarization_output_tokens
                FROM books
                """
            )
            for row in cur.fetchall():
                book = Book(
                    id=row["id"],
                    title=row["title"],
                    author=row["author"],
                    format=BookFormat(row["format"]),
                    file_path=row["file_path"],
                    file_hash=row["file_hash"],
                    total_chunks=row["total_chunks"] or 0,
                    total_pages=row["total_pages"],
                    ingested_at=row["ingested_at"],
                    embedding_tokens=row["embedding_tokens"] or 0,
                    summarization_input_tokens=row["summarization_input_tokens"] or 0,
                    summarization_output_tokens=row["summarization_output_tokens"] or 0,
                )
                self._registry.add_book(book)

        return self._registry

    def save(self) -> None:
        """Save is a no-op since we save immediately on each change."""
        pass

    def add_book(self, book: Book) -> None:
        """Add a book to the registry."""
        with get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO books (
                    id, title, author, format, file_path, file_hash,
                    total_chunks, total_pages, ingested_at,
                    embedding_tokens, summarization_input_tokens,
                    summarization_output_tokens
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    author = EXCLUDED.author,
                    format = EXCLUDED.format,
                    file_path = EXCLUDED.file_path,
                    file_hash = EXCLUDED.file_hash,
                    total_chunks = EXCLUDED.total_chunks,
                    total_pages = EXCLUDED.total_pages,
                    ingested_at = EXCLUDED.ingested_at,
                    embedding_tokens = EXCLUDED.embedding_tokens,
                    summarization_input_tokens = EXCLUDED.summarization_input_tokens,
                    summarization_output_tokens = EXCLUDED.summarization_output_tokens
                """,
                (
                    book.id,
                    book.title,
                    book.author,
                    book.format.value,
                    book.file_path,
                    book.file_hash,
                    book.total_chunks,
                    book.total_pages,
                    book.ingested_at,
                    book.embedding_tokens,
                    book.summarization_input_tokens,
                    book.summarization_output_tokens,
                ),
            )

        # Update in-memory cache if loaded
        if self._registry is not None:
            self._registry.add_book(book)

    def get_book(self, book_id: str) -> Optional[Book]:
        """Get a book by ID."""
        registry = self.load()
        return registry.get_book(book_id)

    def remove_book(self, book_id: str) -> Optional[Book]:
        """Remove a book and return it."""
        registry = self.load()
        book = registry.get_book(book_id)
        if not book:
            return None

        with get_cursor() as cur:
            # Foreign keys will cascade delete chunks, sessions, indices
            cur.execute("DELETE FROM books WHERE id = %s", (book_id,))

        # Update in-memory cache
        if self._registry is not None:
            self._registry.remove_book(book_id)

        return book

    def find_by_hash(self, file_hash: str) -> Optional[Book]:
        """Find a book by file hash."""
        registry = self.load()
        return registry.find_by_hash(file_hash)

    def list_books(self) -> list[Book]:
        """List all books."""
        registry = self.load()
        return registry.list_books()


class PgSessionStore:
    """PostgreSQL storage for chat sessions.

    Drop-in replacement for file-based SessionStore.
    """

    def __init__(self, data_dir=None):
        """Initialize the session store.

        Args:
            data_dir: Ignored. Kept for API compatibility.
        """
        pass

    def save(self, session: Session) -> None:
        """Save a session to database."""
        with get_cursor() as cur:
            # Upsert session
            cur.execute(
                """
                INSERT INTO sessions (id, book_id, provider, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    provider = EXCLUDED.provider,
                    updated_at = EXCLUDED.updated_at
                """,
                (
                    session.id,
                    session.book_id,
                    session.provider,
                    session.created_at,
                    session.updated_at,
                ),
            )

            # Delete existing messages and re-insert all
            cur.execute(
                "DELETE FROM chat_messages WHERE session_id = %s", (session.id,)
            )

            # Insert messages
            for msg in session.messages:
                cur.execute(
                    """
                    INSERT INTO chat_messages (
                        session_id, role, content, citations,
                        input_tokens, output_tokens, timestamp
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        session.id,
                        msg.role.value,
                        msg.content,
                        msg.citations,
                        msg.input_tokens,
                        msg.output_tokens,
                        msg.timestamp,
                    ),
                )

    def load(self, book_id: str, session_id: str) -> Optional[Session]:
        """Load a session from database."""
        with get_cursor(dict_cursor=True) as cur:
            # Get session
            cur.execute(
                """
                SELECT id, book_id, provider, created_at, updated_at
                FROM sessions
                WHERE id = %s AND book_id = %s
                """,
                (session_id, book_id),
            )
            row = cur.fetchone()
            if not row:
                return None

            # Get messages
            cur.execute(
                """
                SELECT role, content, citations, input_tokens,
                       output_tokens, timestamp
                FROM chat_messages
                WHERE session_id = %s
                ORDER BY id ASC
                """,
                (session_id,),
            )
            messages = []
            for msg_row in cur.fetchall():
                messages.append(
                    ChatMessage(
                        role=ChatRole(msg_row["role"]),
                        content=msg_row["content"],
                        citations=msg_row["citations"],
                        input_tokens=msg_row["input_tokens"],
                        output_tokens=msg_row["output_tokens"],
                        timestamp=msg_row["timestamp"],
                    )
                )

            return Session(
                id=row["id"],
                book_id=row["book_id"],
                provider=row["provider"],
                messages=messages,
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    def list_sessions(self, book_id: str) -> list[Session]:
        """List all sessions for a book."""
        sessions = []

        with get_cursor(dict_cursor=True) as cur:
            cur.execute(
                """
                SELECT id FROM sessions
                WHERE book_id = %s
                ORDER BY updated_at DESC
                """,
                (book_id,),
            )
            session_ids = [row["id"] for row in cur.fetchall()]

        for session_id in session_ids:
            session = self.load(book_id, session_id)
            if session:
                sessions.append(session)

        return sessions

    def delete_session(self, book_id: str, session_id: str) -> bool:
        """Delete a session."""
        with get_cursor() as cur:
            cur.execute(
                "DELETE FROM sessions WHERE id = %s AND book_id = %s",
                (session_id, book_id),
            )
            return cur.rowcount > 0

    def delete_book_sessions(self, book_id: str) -> int:
        """Delete all sessions for a book.

        Returns:
            Number of sessions deleted
        """
        with get_cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE book_id = %s", (book_id,))
            return cur.rowcount

    def get_latest_session(self, book_id: str) -> Optional[Session]:
        """Get the most recent session for a book."""
        with get_cursor(dict_cursor=True) as cur:
            cur.execute(
                """
                SELECT id FROM sessions
                WHERE book_id = %s
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (book_id,),
            )
            row = cur.fetchone()
            if not row:
                return None

        return self.load(book_id, row["id"])


class PgBookIndexStore:
    """PostgreSQL storage for book indices (summaries, narratives, navigation).

    Drop-in replacement for file-based BookIndexStore.
    """

    def __init__(self, data_dir=None):
        """Initialize the book index store.

        Args:
            data_dir: Ignored. Kept for API compatibility.
        """
        pass

    def save(self, index: BookIndex) -> None:
        """Save a book index to database."""
        with get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO book_indices (
                    book_id, title, author, book_summary,
                    chapter_summaries, chapter_index, all_narratives,
                    model_used, total_input_tokens, total_output_tokens,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (book_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    author = EXCLUDED.author,
                    book_summary = EXCLUDED.book_summary,
                    chapter_summaries = EXCLUDED.chapter_summaries,
                    chapter_index = EXCLUDED.chapter_index,
                    all_narratives = EXCLUDED.all_narratives,
                    model_used = EXCLUDED.model_used,
                    total_input_tokens = EXCLUDED.total_input_tokens,
                    total_output_tokens = EXCLUDED.total_output_tokens,
                    created_at = EXCLUDED.created_at
                """,
                (
                    index.book_id,
                    index.title,
                    index.author,
                    json.dumps(index.book_summary.model_dump()),
                    json.dumps([cs.model_dump() for cs in index.chapter_summaries]),
                    json.dumps([ci.model_dump() for ci in index.chapter_index]),
                    json.dumps([n.model_dump() for n in index.all_narratives]),
                    index.model_used,
                    index.total_input_tokens,
                    index.total_output_tokens,
                    index.created_at,
                ),
            )

    def load(self, book_id: str) -> Optional[BookIndex]:
        """Load a book index from database."""
        with get_cursor(dict_cursor=True) as cur:
            cur.execute(
                """
                SELECT book_id, title, author, book_summary,
                       chapter_summaries, chapter_index, all_narratives,
                       model_used, total_input_tokens, total_output_tokens,
                       created_at
                FROM book_indices
                WHERE book_id = %s
                """,
                (book_id,),
            )
            row = cur.fetchone()
            if not row:
                return None

        # Parse JSONB fields back to Pydantic models
        book_summary_data = row["book_summary"]
        if isinstance(book_summary_data, str):
            book_summary_data = json.loads(book_summary_data)
        book_summary = BookSummary.model_validate(book_summary_data)

        chapter_summaries_data = row["chapter_summaries"]
        if isinstance(chapter_summaries_data, str):
            chapter_summaries_data = json.loads(chapter_summaries_data)
        chapter_summaries = [
            ChapterSummary.model_validate(cs) for cs in chapter_summaries_data
        ]

        chapter_index_data = row["chapter_index"]
        if isinstance(chapter_index_data, str):
            chapter_index_data = json.loads(chapter_index_data)
        chapter_index = [
            ChapterIndexEntry.model_validate(ci) for ci in chapter_index_data
        ]

        all_narratives_data = row["all_narratives"]
        if isinstance(all_narratives_data, str):
            all_narratives_data = json.loads(all_narratives_data)
        all_narratives = [Narrative.model_validate(n) for n in all_narratives_data]

        return BookIndex(
            book_id=row["book_id"],
            title=row["title"],
            author=row["author"],
            book_summary=book_summary,
            chapter_summaries=chapter_summaries,
            chapter_index=chapter_index,
            all_narratives=all_narratives,
            model_used=row["model_used"],
            total_input_tokens=row["total_input_tokens"] or 0,
            total_output_tokens=row["total_output_tokens"] or 0,
            created_at=row["created_at"],
        )

    def exists(self, book_id: str) -> bool:
        """Check if an index exists for a book."""
        with get_cursor() as cur:
            cur.execute(
                "SELECT 1 FROM book_indices WHERE book_id = %s", (book_id,)
            )
            return cur.fetchone() is not None

    def delete(self, book_id: str) -> bool:
        """Delete a book index."""
        with get_cursor() as cur:
            cur.execute("DELETE FROM book_indices WHERE book_id = %s", (book_id,))
            return cur.rowcount > 0
