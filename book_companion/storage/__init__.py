"""Storage backends for vectors, sessions, and book indices.

This module provides factory functions that automatically select the appropriate
backend based on environment configuration:

- If DATABASE_URL or CLOUD_SQL_CONNECTION_NAME is set: PostgreSQL backends
- Otherwise: File-based backends (ChromaDB + JSON)

Usage:
    from book_companion.storage import (
        get_vector_store,
        get_book_registry_store,
        get_session_store,
        get_book_index_store,
    )

    vector_store = get_vector_store()
    registry = get_book_registry_store()
"""

from pathlib import Path
from typing import Optional, Union

from .vector_store import VectorStore, get_data_dir
from .session_store import SessionStore, BookRegistryStore, BookIndexStore

# Type aliases for factory return types
VectorStoreType = Union[VectorStore, "PgVectorStore"]
BookRegistryStoreType = Union[BookRegistryStore, "PgBookRegistryStore"]
SessionStoreType = Union[SessionStore, "PgSessionStore"]
BookIndexStoreType = Union[BookIndexStore, "PgBookIndexStore"]


def _use_postgres() -> bool:
    """Check if PostgreSQL should be used based on environment."""
    from .database import is_postgres_configured

    return is_postgres_configured()


def get_vector_store(persist_dir: Optional[Path] = None) -> VectorStoreType:
    """Get a vector store instance.

    Returns PostgreSQL-backed store if configured, otherwise ChromaDB.

    Args:
        persist_dir: Directory for ChromaDB persistence (ignored for PostgreSQL).

    Returns:
        VectorStore or PgVectorStore instance.
    """
    if _use_postgres():
        from .pg_vector_store import PgVectorStore

        return PgVectorStore()
    return VectorStore(persist_dir=persist_dir)


def get_book_registry_store(data_dir: Optional[Path] = None) -> BookRegistryStoreType:
    """Get a book registry store instance.

    Returns PostgreSQL-backed store if configured, otherwise file-based.

    Args:
        data_dir: Data directory for file storage (ignored for PostgreSQL).

    Returns:
        BookRegistryStore or PgBookRegistryStore instance.
    """
    if _use_postgres():
        from .pg_session_store import PgBookRegistryStore

        return PgBookRegistryStore()
    return BookRegistryStore(data_dir=data_dir)


def get_session_store(data_dir: Optional[Path] = None) -> SessionStoreType:
    """Get a session store instance.

    Returns PostgreSQL-backed store if configured, otherwise file-based.

    Args:
        data_dir: Data directory for file storage (ignored for PostgreSQL).

    Returns:
        SessionStore or PgSessionStore instance.
    """
    if _use_postgres():
        from .pg_session_store import PgSessionStore

        return PgSessionStore()
    return SessionStore(data_dir=data_dir)


def get_book_index_store(data_dir: Optional[Path] = None) -> BookIndexStoreType:
    """Get a book index store instance.

    Returns PostgreSQL-backed store if configured, otherwise file-based.

    Args:
        data_dir: Data directory for file storage (ignored for PostgreSQL).

    Returns:
        BookIndexStore or PgBookIndexStore instance.
    """
    if _use_postgres():
        from .pg_session_store import PgBookIndexStore

        return PgBookIndexStore()
    return BookIndexStore(data_dir=data_dir)


def init_storage() -> None:
    """Initialize storage backend.

    For PostgreSQL: Creates tables if they don't exist.
    For file-based: Creates directories if they don't exist.
    """
    if _use_postgres():
        from .database import init_schema

        init_schema()
    else:
        # File-based stores create directories on init
        get_data_dir()


# Direct imports for backward compatibility
# These classes can still be imported and used directly
__all__ = [
    # File-based backends (original)
    "VectorStore",
    "SessionStore",
    "BookRegistryStore",
    "BookIndexStore",
    "get_data_dir",
    # Factory functions (recommended)
    "get_vector_store",
    "get_book_registry_store",
    "get_session_store",
    "get_book_index_store",
    "init_storage",
]
