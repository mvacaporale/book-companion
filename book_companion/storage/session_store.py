"""Session, book registry, and book index storage."""

import json
from pathlib import Path
from typing import Optional

from book_companion.models import Book, BookIndex, BookRegistry, Session
from .vector_store import get_data_dir


class BookRegistryStore:
    """Persistent storage for the book registry."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the book registry store.

        Args:
            data_dir: Data directory. Defaults to ~/.bookrc/
        """
        self.data_dir = data_dir or get_data_dir()
        self.registry_path = self.data_dir / "books.json"
        self._registry: Optional[BookRegistry] = None

    def load(self) -> BookRegistry:
        """Load the registry from disk."""
        if self._registry is not None:
            return self._registry

        if self.registry_path.exists():
            try:
                data = json.loads(self.registry_path.read_text())
                self._registry = BookRegistry.model_validate(data)
            except Exception:
                self._registry = BookRegistry()
        else:
            self._registry = BookRegistry()

        return self._registry

    def save(self) -> None:
        """Save the registry to disk."""
        if self._registry is None:
            return

        self.registry_path.write_text(
            self._registry.model_dump_json(indent=2)
        )

    def add_book(self, book: Book) -> None:
        """Add a book to the registry and save."""
        registry = self.load()
        registry.add_book(book)
        self.save()

    def get_book(self, book_id: str) -> Optional[Book]:
        """Get a book by ID."""
        registry = self.load()
        return registry.get_book(book_id)

    def remove_book(self, book_id: str) -> Optional[Book]:
        """Remove a book and save."""
        registry = self.load()
        book = registry.remove_book(book_id)
        if book:
            self.save()
        return book

    def find_by_hash(self, file_hash: str) -> Optional[Book]:
        """Find a book by file hash."""
        registry = self.load()
        return registry.find_by_hash(file_hash)

    def list_books(self) -> list[Book]:
        """List all books."""
        registry = self.load()
        return registry.list_books()


class SessionStore:
    """Persistent storage for chat sessions."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the session store.

        Args:
            data_dir: Data directory. Defaults to ~/.bookrc/
        """
        self.data_dir = data_dir or get_data_dir()
        self.sessions_dir = self.data_dir / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _get_book_sessions_dir(self, book_id: str) -> Path:
        """Get the sessions directory for a book."""
        book_dir = self.sessions_dir / book_id
        book_dir.mkdir(parents=True, exist_ok=True)
        return book_dir

    def _get_session_path(self, book_id: str, session_id: str) -> Path:
        """Get the path for a session file."""
        return self._get_book_sessions_dir(book_id) / f"{session_id}.json"

    def save(self, session: Session) -> None:
        """Save a session to disk."""
        path = self._get_session_path(session.book_id, session.id)
        path.write_text(session.model_dump_json(indent=2))

    def load(self, book_id: str, session_id: str) -> Optional[Session]:
        """Load a session from disk."""
        path = self._get_session_path(book_id, session_id)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            return Session.model_validate(data)
        except Exception:
            return None

    def list_sessions(self, book_id: str) -> list[Session]:
        """List all sessions for a book."""
        book_dir = self._get_book_sessions_dir(book_id)
        sessions = []

        for path in book_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                session = Session.model_validate(data)
                sessions.append(session)
            except Exception:
                continue

        # Sort by updated_at, most recent first
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def delete_session(self, book_id: str, session_id: str) -> bool:
        """Delete a session."""
        path = self._get_session_path(book_id, session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def delete_book_sessions(self, book_id: str) -> int:
        """Delete all sessions for a book.

        Returns:
            Number of sessions deleted
        """
        book_dir = self._get_book_sessions_dir(book_id)
        count = 0
        for path in book_dir.glob("*.json"):
            path.unlink()
            count += 1

        # Remove the directory if empty
        try:
            book_dir.rmdir()
        except Exception:
            pass

        return count

    def get_latest_session(self, book_id: str) -> Optional[Session]:
        """Get the most recent session for a book."""
        sessions = self.list_sessions(book_id)
        return sessions[0] if sessions else None


class BookIndexStore:
    """Persistent storage for book indices (summaries, narratives, navigation)."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the book index store.

        Args:
            data_dir: Data directory. Defaults to ~/.bookrc/
        """
        self.data_dir = data_dir or get_data_dir()
        self.indices_dir = self.data_dir / "indices"
        self.indices_dir.mkdir(parents=True, exist_ok=True)

    def _get_index_path(self, book_id: str) -> Path:
        """Get the path for a book index file."""
        return self.indices_dir / f"{book_id}.json"

    def save(self, index: BookIndex) -> None:
        """Save a book index to disk.

        Args:
            index: The BookIndex to save
        """
        path = self._get_index_path(index.book_id)
        path.write_text(index.model_dump_json(indent=2))

    def load(self, book_id: str) -> Optional[BookIndex]:
        """Load a book index from disk.

        Args:
            book_id: The book ID

        Returns:
            BookIndex if found, None otherwise
        """
        path = self._get_index_path(book_id)
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text())
            return BookIndex.model_validate(data)
        except Exception:
            return None

    def exists(self, book_id: str) -> bool:
        """Check if an index exists for a book.

        Args:
            book_id: The book ID

        Returns:
            True if index exists
        """
        return self._get_index_path(book_id).exists()

    def delete(self, book_id: str) -> bool:
        """Delete a book index.

        Args:
            book_id: The book ID

        Returns:
            True if deleted
        """
        path = self._get_index_path(book_id)
        if path.exists():
            path.unlink()
            return True
        return False
