"""PgVectorStore.query drops junk and returns distances (DB faked)."""

from contextlib import contextmanager

from book_companion.storage import pg_vector_store
from book_companion.storage.pg_vector_store import PgVectorStore

PROSE = "word " * 40


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.params = None

    def execute(self, sql, params):
        self.params = params

    def fetchall(self):
        return self.rows


def test_query_filters_junk_and_overfetches(monkeypatch) -> None:
    rows = [
        {"id": "a", "text": PROSE, "chapter_title": "Index", "chapter_number": 20,
         "page_number": 300, "start_char": 0, "end_char": 1, "distance": 0.1},
        {"id": "b", "text": "PART TWO", "chapter_title": "Part Two", "chapter_number": 5,
         "page_number": 40, "start_char": 0, "end_char": 1, "distance": 0.2},
        {"id": "c", "text": PROSE, "chapter_title": "CHAPTER 3", "chapter_number": 3,
         "page_number": 41, "start_char": 0, "end_char": 1, "distance": 0.3},
        {"id": "d", "text": PROSE, "chapter_title": "CHAPTER 4", "chapter_number": 4,
         "page_number": 60, "start_char": 0, "end_char": 1, "distance": 0.4},
    ]
    cursor = FakeCursor(rows)

    @contextmanager
    def fake_get_cursor(dict_cursor=False):
        yield cursor

    monkeypatch.setattr(pg_vector_store, "get_cursor", fake_get_cursor)

    result = PgVectorStore().query("book", [0.0, 1.0], n_results=1)
    assert cursor.params[-1] == 1 * pg_vector_store.OVERFETCH
    assert result.chunk_ids == ["c"]
    assert result.distances == [0.3]
