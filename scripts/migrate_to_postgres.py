#!/usr/bin/env python3
"""Migration script: ChromaDB + JSON to PostgreSQL + pgvector.

This script exports data from the existing file-based storage (ChromaDB + JSON)
to a portable format, then imports it into PostgreSQL with pgvector.

Usage:
    # Export from local file-based storage
    python scripts/migrate_to_postgres.py export --output-dir ./migration_export

    # Export from a specific data directory (e.g., downloaded from GCS)
    python scripts/migrate_to_postgres.py export --data-dir ./migration_data --output-dir ./migration_export

    # Import to PostgreSQL (requires DATABASE_URL or Cloud SQL config)
    DATABASE_URL=postgresql://user:pass@localhost/bookcompanion \\
        python scripts/migrate_to_postgres.py import --input-dir ./migration_export

    # Verify migration
    DATABASE_URL=postgresql://user:pass@localhost/bookcompanion \\
        python scripts/migrate_to_postgres.py verify
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def export_data(data_dir: Optional[Path], output_dir: Path) -> None:
    """Export data from ChromaDB and JSON files to portable JSON format.

    Args:
        data_dir: Source data directory (default: ~/.bookrc/)
        output_dir: Directory to write exported JSON files
    """
    import chromadb
    from chromadb.config import Settings

    from book_companion.storage.vector_store import get_data_dir
    from book_companion.storage.session_store import (
        BookRegistryStore,
        SessionStore,
        BookIndexStore,
    )

    if data_dir is None:
        data_dir = get_data_dir()
    else:
        data_dir = Path(data_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting from: {data_dir}")
    print(f"Exporting to: {output_dir}")

    # 1. Export books.json
    registry_store = BookRegistryStore(data_dir=data_dir)
    books = registry_store.list_books()
    books_data = [book.model_dump(mode="json") for book in books]
    (output_dir / "books.json").write_text(json.dumps(books_data, indent=2))
    print(f"Exported {len(books)} books")

    # 2. Export sessions
    session_store = SessionStore(data_dir=data_dir)
    all_sessions = []
    for book in books:
        sessions = session_store.list_sessions(book.id)
        for session in sessions:
            all_sessions.append(session.model_dump(mode="json"))
    (output_dir / "sessions.json").write_text(json.dumps(all_sessions, indent=2))
    print(f"Exported {len(all_sessions)} sessions")

    # 3. Export book indices
    index_store = BookIndexStore(data_dir=data_dir)
    all_indices = []
    for book in books:
        index = index_store.load(book.id)
        if index:
            all_indices.append(index.model_dump(mode="json"))
    (output_dir / "indices.json").write_text(json.dumps(all_indices, indent=2))
    print(f"Exported {len(all_indices)} book indices")

    # 4. Export chunks with embeddings from ChromaDB
    db_path = data_dir / "db"
    if not db_path.exists():
        print(f"Warning: ChromaDB directory not found at {db_path}")
        (output_dir / "chunks.json").write_text("[]")
        return

    client = chromadb.PersistentClient(
        path=str(db_path),
        settings=Settings(anonymized_telemetry=False),
    )

    all_chunks = []
    total_chunks = 0

    for book in books:
        collection_name = f"book_{book.id}"
        try:
            collection = client.get_collection(name=collection_name)
            count = collection.count()
            if count == 0:
                continue

            # Get all data from collection
            result = collection.get(
                include=["documents", "metadatas", "embeddings"],
                limit=count,
            )

            for i, chunk_id in enumerate(result["ids"]):
                # Handle numpy arrays from ChromaDB
                embedding = None
                if result["embeddings"] is not None and len(result["embeddings"]) > i:
                    emb = result["embeddings"][i]
                    # Convert numpy array to list if needed
                    if hasattr(emb, "tolist"):
                        embedding = emb.tolist()
                    else:
                        embedding = list(emb) if emb is not None else None

                chunk_data = {
                    "id": chunk_id,
                    "book_id": book.id,
                    "text": result["documents"][i],
                    "embedding": embedding,
                    "metadata": result["metadatas"][i],
                }
                all_chunks.append(chunk_data)
                total_chunks += 1

            print(f"  Book '{book.title}': {count} chunks")

        except Exception as e:
            print(f"  Warning: Could not export book {book.id}: {e}")

    # Write chunks to file (may be large)
    (output_dir / "chunks.json").write_text(json.dumps(all_chunks))
    print(f"Exported {total_chunks} total chunks")


def import_data(input_dir: Path) -> None:
    """Import data from portable JSON format to PostgreSQL.

    Args:
        input_dir: Directory containing exported JSON files
    """
    from book_companion.models import Book, BookIndex, Session
    from book_companion.storage.database import init_schema, get_cursor
    from book_companion.storage.pg_session_store import (
        PgBookRegistryStore,
        PgSessionStore,
        PgBookIndexStore,
    )
    from book_companion.storage.pg_vector_store import PgVectorStore
    from book_companion.models import Chunk, ChunkMetadata

    input_dir = Path(input_dir)

    print(f"Importing from: {input_dir}")
    print("Initializing database schema...")
    init_schema()

    # 1. Import books
    books_file = input_dir / "books.json"
    if books_file.exists():
        books_data = json.loads(books_file.read_text())
        registry = PgBookRegistryStore()
        for book_data in books_data:
            book = Book.model_validate(book_data)
            registry.add_book(book)
        print(f"Imported {len(books_data)} books")
    else:
        print("Warning: books.json not found")
        books_data = []

    # 2. Import chunks with embeddings
    chunks_file = input_dir / "chunks.json"
    if chunks_file.exists():
        chunks_data = json.loads(chunks_file.read_text())
        vector_store = PgVectorStore()

        # Group chunks by book_id for batch insertion
        chunks_by_book: dict[str, list[Chunk]] = {}
        for chunk_data in chunks_data:
            book_id = chunk_data["book_id"]
            metadata = chunk_data["metadata"]

            chunk = Chunk(
                id=chunk_data["id"],
                text=chunk_data["text"],
                embedding=chunk_data.get("embedding"),
                metadata=ChunkMetadata(
                    book_id=book_id,
                    chapter_title=metadata.get("chapter_title") or None,
                    chapter_number=metadata.get("chapter_number") or None,
                    page_number=metadata.get("page_number") or None,
                    start_char=metadata.get("start_char", 0),
                    end_char=metadata.get("end_char", 0),
                ),
            )

            if book_id not in chunks_by_book:
                chunks_by_book[book_id] = []
            chunks_by_book[book_id].append(chunk)

        for book_id, chunks in chunks_by_book.items():
            vector_store.add_chunks(book_id, chunks)
            print(f"  Book {book_id}: {len(chunks)} chunks")

        print(f"Imported {len(chunks_data)} total chunks")
    else:
        print("Warning: chunks.json not found")

    # 3. Import sessions
    sessions_file = input_dir / "sessions.json"
    if sessions_file.exists():
        sessions_data = json.loads(sessions_file.read_text())
        session_store = PgSessionStore()
        for session_data in sessions_data:
            session = Session.model_validate(session_data)
            session_store.save(session)
        print(f"Imported {len(sessions_data)} sessions")
    else:
        print("Warning: sessions.json not found")

    # 4. Import book indices
    indices_file = input_dir / "indices.json"
    if indices_file.exists():
        indices_data = json.loads(indices_file.read_text())
        index_store = PgBookIndexStore()
        for index_data in indices_data:
            index = BookIndex.model_validate(index_data)
            index_store.save(index)
        print(f"Imported {len(indices_data)} book indices")
    else:
        print("Warning: indices.json not found")

    print("\nMigration complete!")


def verify_migration() -> None:
    """Verify migration by checking counts in PostgreSQL."""
    from book_companion.storage.database import get_cursor

    print("Verifying PostgreSQL data...")

    with get_cursor() as cur:
        # Count books
        cur.execute("SELECT COUNT(*) FROM books")
        book_count = cur.fetchone()[0]

        # Count chunks
        cur.execute("SELECT COUNT(*) FROM chunks")
        chunk_count = cur.fetchone()[0]

        # Count sessions
        cur.execute("SELECT COUNT(*) FROM sessions")
        session_count = cur.fetchone()[0]

        # Count book indices
        cur.execute("SELECT COUNT(*) FROM book_indices")
        index_count = cur.fetchone()[0]

        # Get per-book chunk counts
        cur.execute("""
            SELECT b.title, COUNT(c.id) as chunk_count
            FROM books b
            LEFT JOIN chunks c ON b.id = c.book_id
            GROUP BY b.id, b.title
            ORDER BY b.title
        """)
        book_chunks = cur.fetchall()

    print(f"\nBooks: {book_count}")
    print(f"Chunks: {chunk_count}")
    print(f"Sessions: {session_count}")
    print(f"Book indices: {index_count}")

    print("\nChunks per book:")
    for title, count in book_chunks:
        print(f"  {title}: {count}")

    print(f"\nTotal: {book_count} books, {chunk_count} chunks, {session_count} sessions, {index_count} indices")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate book-companion data from ChromaDB to PostgreSQL"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Export command
    export_parser = subparsers.add_parser(
        "export", help="Export data from ChromaDB/JSON to portable format"
    )
    export_parser.add_argument(
        "--data-dir",
        type=Path,
        help="Source data directory (default: ~/.bookrc/)",
    )
    export_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for exported JSON files",
    )

    # Import command
    import_parser = subparsers.add_parser(
        "import", help="Import data from portable format to PostgreSQL"
    )
    import_parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing exported JSON files",
    )

    # Verify command
    subparsers.add_parser(
        "verify", help="Verify migration by checking PostgreSQL counts"
    )

    args = parser.parse_args()

    if args.command == "export":
        export_data(args.data_dir, args.output_dir)
    elif args.command == "import":
        import_data(args.input_dir)
    elif args.command == "verify":
        verify_migration()


if __name__ == "__main__":
    main()
