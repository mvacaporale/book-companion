"""bookrc edit fixes metadata in place (file-based stores)."""

from click.testing import CliRunner

from book_companion.cli import cli
from book_companion.models import Book, BookFormat
from book_companion.storage import get_book_registry_store


def test_edit_updates_title_and_author(mock_bookrc_dir, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    registry = get_book_registry_store()
    book = Book(title="[No data]", author="443186283", format=BookFormat.EPUB, file_path="x", file_hash="h")
    registry.add_book(book)

    result = CliRunner().invoke(cli, ["edit", book.id, "-t", "Affluence Without Abundance", "-a", "James Suzman"])

    assert result.exit_code == 0, result.output
    updated = get_book_registry_store().get_book(book.id)
    assert (updated.title, updated.author) == ("Affluence Without Abundance", "James Suzman")
