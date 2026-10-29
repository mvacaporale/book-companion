"""Re-ingesting keeps the book ID and its chat sessions (file-based stores)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from book_companion import ingestion
from book_companion.storage import get_book_registry_store, get_session_store


class FakeEmbeddings:
    def embed_chunks(self, chunks):
        for chunk in chunks:
            chunk.embedding = [0.1, 0.2, 0.3]
        return chunks


@pytest.fixture
def file_storage(mock_bookrc_dir, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr(Path, "home", lambda: mock_bookrc_dir.parent)
    return mock_bookrc_dir


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_replace_book_id_keeps_id_and_sessions(file_storage, temp_dir):
    body = "Some real prose about minds and bodies. " * 20
    first = _write(temp_dir / "situated.md", f"# Situated\n\n## Foreword\n\n{body}\n\n## Chapter 1\n\n{body}\n")
    with patch.object(ingestion, "EmbeddingClient", FakeEmbeddings):
        book_id = ingestion.ingest_book(first, skip_summary=True)["book_id"]

        from book_companion.models import Session
        get_session_store().save(Session(book_id=book_id))

        # A different copy of the file (different hash), replaced in place
        second = _write(temp_dir / "situated-v2.md", f"# Situated\n\n## Chapter 1\n\n{body}\n\n## Chapter 2\n\n{body}\n")
        result = ingestion.ingest_book(second, skip_summary=True, replace_book_id=book_id)

    assert result["book_id"] == book_id
    books = get_book_registry_store().list_books()
    assert [b.id for b in books] == [book_id]
    assert get_session_store().list_sessions(book_id)


def test_replace_refuses_a_file_owned_by_another_book(file_storage, temp_dir):
    body = "Some real prose about minds and bodies. " * 20
    a = _write(temp_dir / "a.md", f"# A\n\n## One\n\n{body}\n\n## Two\n\n{body}\n")
    b = _write(temp_dir / "b.md", f"# B\n\n## One\n\n{body} b\n\n## Two\n\n{body}\n")
    with patch.object(ingestion, "EmbeddingClient", FakeEmbeddings):
        id_a = ingestion.ingest_book(a, skip_summary=True)["book_id"]
        ingestion.ingest_book(b, skip_summary=True)
        with pytest.raises(ValueError, match="already ingested"):
            ingestion.ingest_book(b, skip_summary=True, replace_book_id=id_a)
