"""PostgreSQL database connection management for Cloud SQL with pgvector."""

import os
from contextlib import contextmanager
from typing import Generator, Optional

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor


# Global connection pool
_pool: Optional[pool.ThreadedConnectionPool] = None


def get_database_url() -> Optional[str]:
    """Get database URL from environment.

    Returns:
        Database URL string, or None if not configured.
    """
    # Direct URL takes precedence
    if url := os.getenv("DATABASE_URL"):
        return url

    # Cloud SQL configuration
    connection_name = os.getenv("CLOUD_SQL_CONNECTION_NAME")
    if connection_name:
        db_name = os.getenv("DB_NAME", "bookcompanion")
        db_user = os.getenv("DB_USER", "bookcompanion")
        db_password = os.getenv("DB_PASSWORD", "")

        # Cloud Run uses Unix socket path
        socket_dir = f"/cloudsql/{connection_name}"
        return f"postgresql://{db_user}:{db_password}@/{db_name}?host={socket_dir}"

    return None


def is_postgres_configured() -> bool:
    """Check if PostgreSQL is configured via environment variables."""
    return get_database_url() is not None


def get_pool() -> pool.ThreadedConnectionPool:
    """Get or create the connection pool.

    Returns:
        ThreadedConnectionPool instance.

    Raises:
        RuntimeError: If database is not configured.
    """
    global _pool

    if _pool is not None:
        return _pool

    url = get_database_url()
    if not url:
        raise RuntimeError(
            "PostgreSQL not configured. Set DATABASE_URL or "
            "CLOUD_SQL_CONNECTION_NAME environment variable."
        )

    _pool = pool.ThreadedConnectionPool(
        minconn=1,
        maxconn=10,
        dsn=url,
    )
    return _pool


@contextmanager
def get_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """Get a database connection from the pool.

    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ...")

    Yields:
        A database connection that auto-commits on success.
    """
    pool = get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


@contextmanager
def get_cursor(
    dict_cursor: bool = False,
) -> Generator[psycopg2.extensions.cursor, None, None]:
    """Get a database cursor with automatic connection management.

    Args:
        dict_cursor: If True, return rows as dictionaries.

    Usage:
        with get_cursor() as cur:
            cur.execute("SELECT * FROM books")
            rows = cur.fetchall()

    Yields:
        A database cursor.
    """
    cursor_factory = RealDictCursor if dict_cursor else None
    with get_connection() as conn:
        with conn.cursor(cursor_factory=cursor_factory) as cur:
            yield cur


def init_schema() -> None:
    """Initialize the database schema with all required tables.

    Safe to call multiple times - uses IF NOT EXISTS.
    """
    schema_sql = """
    -- Enable pgvector extension
    CREATE EXTENSION IF NOT EXISTS vector;

    -- Books table (replaces books.json)
    CREATE TABLE IF NOT EXISTS books (
        id VARCHAR(8) PRIMARY KEY,
        title TEXT NOT NULL,
        author TEXT,
        format VARCHAR(20) NOT NULL,
        file_path TEXT NOT NULL,
        file_hash VARCHAR(32) NOT NULL UNIQUE,
        total_chunks INTEGER DEFAULT 0,
        total_pages INTEGER,
        ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        embedding_tokens INTEGER DEFAULT 0,
        summarization_input_tokens INTEGER DEFAULT 0,
        summarization_output_tokens INTEGER DEFAULT 0
    );

    -- Chunks with vector embeddings (replaces ChromaDB)
    CREATE TABLE IF NOT EXISTS chunks (
        id VARCHAR(8) PRIMARY KEY,
        book_id VARCHAR(8) REFERENCES books(id) ON DELETE CASCADE,
        text TEXT NOT NULL,
        embedding vector(768) NOT NULL,
        chapter_title TEXT,
        chapter_number INTEGER,
        page_number INTEGER,
        start_char INTEGER NOT NULL,
        end_char INTEGER NOT NULL
    );

    -- Create HNSW index for fast ANN search if not exists
    DO $$
    BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_indexes
            WHERE indexname = 'idx_chunks_embedding'
        ) THEN
            CREATE INDEX idx_chunks_embedding ON chunks
            USING hnsw (embedding vector_cosine_ops);
        END IF;
    END $$;

    -- Index for filtering by book
    CREATE INDEX IF NOT EXISTS idx_chunks_book_id ON chunks(book_id);

    -- Sessions table
    CREATE TABLE IF NOT EXISTS sessions (
        id VARCHAR(8) PRIMARY KEY,
        book_id VARCHAR(8) REFERENCES books(id) ON DELETE CASCADE,
        provider VARCHAR(20) DEFAULT 'gemini',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Index for listing sessions by book
    CREATE INDEX IF NOT EXISTS idx_sessions_book_id ON sessions(book_id);

    -- Chat messages table
    CREATE TABLE IF NOT EXISTS chat_messages (
        id SERIAL PRIMARY KEY,
        session_id VARCHAR(8) REFERENCES sessions(id) ON DELETE CASCADE,
        role VARCHAR(10) NOT NULL,
        content TEXT NOT NULL,
        citations TEXT[],
        input_tokens INTEGER,
        output_tokens INTEGER,
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Index for loading messages by session
    CREATE INDEX IF NOT EXISTS idx_messages_session_id ON chat_messages(session_id);

    -- Book indices table (stores BookIndex as JSONB)
    CREATE TABLE IF NOT EXISTS book_indices (
        book_id VARCHAR(8) PRIMARY KEY REFERENCES books(id) ON DELETE CASCADE,
        title TEXT NOT NULL,
        author TEXT,
        book_summary JSONB NOT NULL,
        chapter_summaries JSONB DEFAULT '[]',
        chapter_index JSONB DEFAULT '[]',
        all_narratives JSONB DEFAULT '[]',
        model_used VARCHAR(50),
        total_input_tokens INTEGER DEFAULT 0,
        total_output_tokens INTEGER DEFAULT 0,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    """

    with get_cursor() as cur:
        cur.execute(schema_sql)


def close_pool() -> None:
    """Close all connections in the pool."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
