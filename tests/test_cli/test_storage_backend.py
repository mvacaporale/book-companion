"""CLI commands must use the configured storage backend (Neon when DATABASE_URL is set)."""

from unittest.mock import MagicMock

from click.testing import CliRunner

from book_companion import cli
from book_companion.models import Book, BookFormat


def test_list_books_reads_from_configured_registry(monkeypatch):
    registry = MagicMock()
    registry.list_books.return_value = [
        Book(id="neon1234", title="Palaces for the People", format=BookFormat.EPUB,
             file_path="x", file_hash="h")
    ]
    monkeypatch.setattr(cli, "get_book_registry_store", lambda: registry)
    monkeypatch.setattr(cli, "get_book_index_store", lambda: MagicMock(exists=lambda _: False))

    result = CliRunner().invoke(cli.cli, ["list", "books"])

    assert result.exit_code == 0, result.output
    assert "neon1234" in result.output
    registry.list_books.assert_called_once()
