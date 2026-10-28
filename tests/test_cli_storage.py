"""CLI commands must use the configured storage backend, not hardcoded file stores."""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from book_companion import cli


def test_list_books_uses_storage_factory():
    registry = MagicMock()
    registry.list_books.return_value = []

    with patch.object(cli, "get_book_registry_store", return_value=registry) as factory:
        result = CliRunner().invoke(cli.cli, ["list", "books"])

    assert result.exit_code == 0, result.output
    factory.assert_called_once()
    registry.list_books.assert_called_once()


def test_cli_has_no_direct_file_store_construction():
    source = open(cli.__file__).read()
    for cls in ("VectorStore(", "BookRegistryStore(", "SessionStore(", "BookIndexStore("):
        assert cls not in source, f"cli.py constructs {cls} directly; use get_*_store()"
