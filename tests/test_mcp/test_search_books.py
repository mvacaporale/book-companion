"""search_books: one overall limit, best matches first."""

import json
from datetime import datetime

import pytest

from book_companion.mcp import server
from book_companion.models import Book, BookFormat, Chunk, ChunkMetadata, RetrievedContext


class FakeRegistry:
    def __init__(self, books):
        self._books = books

    def list_books(self):
        return self._books

    def get_book(self, book_id):
        return next((b for b in self._books if b.id == book_id), None)


class FakeVectorStore:
    """Every book returns n_results chunks; distances differ per book."""

    def __init__(self, distance_by_book):
        self.distance_by_book = distance_by_book

    def query(self, book_id, query_embedding, n_results=8):
        base = self.distance_by_book[book_id]
        chunks = [
            Chunk(id=f"{book_id}-{i}", text=f"text {i}", metadata=ChunkMetadata(book_id=book_id, start_char=0, end_char=1))
            for i in range(n_results)
        ]
        return RetrievedContext(
            chunks=chunks,
            formatted_context="",
            chunk_ids=[c.id for c in chunks],
            distances=[base + i * 0.01 for i in range(n_results)],
        )


class FakeEmbeddings:
    def embed_query(self, query):
        return [0.0]


def _book(book_id):
    return Book(id=book_id, title=book_id, file_path=f"/{book_id}.pdf", file_hash=book_id,
                format=BookFormat.PDF, ingested_at=datetime.now(), total_chunks=10)


@pytest.fixture
def fake_context(monkeypatch):
    ids = [f"book{i}" for i in range(16)]
    ctx = {
        "registry": FakeRegistry([_book(i) for i in ids]),
        "vector_store": FakeVectorStore({b: 0.5 - n * 0.02 for n, b in enumerate(ids)}),
        "embedding_client": FakeEmbeddings(),
    }
    monkeypatch.setattr(server, "get_context", lambda: ctx)
    return ctx


async def test_limit_is_overall_not_per_book(fake_context) -> None:
    results = json.loads(await server.search_books("affluence", n_results=2))
    assert len(results) == 2
    # The two closest chunks overall, best first
    assert [r["chunk_id"] for r in results] == ["book15-0", "book15-1"]
    assert results[0]["score"] >= results[1]["score"]
