"""Tests for GoogleDriveClient file listing."""

from unittest.mock import MagicMock, patch

from book_companion.google_drive.client import GoogleDriveClient


def _client_with_mock_service() -> tuple[GoogleDriveClient, MagicMock]:
    client = GoogleDriveClient()
    service = MagicMock()
    service.files.return_value.list.return_value.execute.return_value = {"files": []}
    client._service = service
    return client, service


def _query_sent(service: MagicMock) -> str:
    return service.files.return_value.list.call_args.kwargs["q"]


class TestListBookFiles:
    """Listing must stay inside the configured books folder."""

    def test_no_folder_arg_uses_configured_default_folder(self):
        client, service = _client_with_mock_service()
        with patch(
            "book_companion.google_drive.client.get_config",
            return_value={"default_folder_id": "books-folder"},
        ):
            client._list_book_files()
        assert "'books-folder' in parents" in _query_sent(service)

    def test_explicit_folder_overrides_default(self):
        client, service = _client_with_mock_service()
        with patch(
            "book_companion.google_drive.client.get_config",
            return_value={"default_folder_id": "books-folder"},
        ):
            client._list_book_files("other-folder")
        query = _query_sent(service)
        assert "'other-folder' in parents" in query
        assert "books-folder" not in query
