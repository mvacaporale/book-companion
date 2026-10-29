"""Tests for automatic Drive folder ingestion."""

from pathlib import Path
from unittest.mock import MagicMock

from book_companion.drive_sync import local_filename, sync_drive_folder
from book_companion.google_drive.client import DriveFile, GoogleDriveClient
from book_companion.models import Book, BookFormat


def _book(title: str, file_hash: str) -> Book:
    return Book(title=title, format=BookFormat.PDF, file_path="x", file_hash=file_hash)


def _drive_file(name: str, md5: str, mime: str = "application/pdf") -> DriveFile:
    return DriveFile(id=f"id-{name}", name=name, mime_type=mime, md5_checksum=md5)


def _client(files: list[DriveFile]) -> GoogleDriveClient:
    client = GoogleDriveClient()
    client._list_book_files = MagicMock(return_value=files)
    client.download_file = MagicMock(side_effect=lambda file_id, dest: dest)
    return client


def _ingest_ok(path: Path, **_):
    return {"book_id": path.stem, "title": path.stem}


class TestSyncDriveFolder:
    def test_skips_books_matched_by_checksum(self, temp_dir):
        client = _client([_drive_file("Unrelated Name.pdf", "abc")])
        ingest = MagicMock()
        result = sync_drive_folder(client, [_book("Atomic Habits", "abc")], ingest, temp_dir)
        assert result.already_ingested == 1
        ingest.assert_not_called()
        client.download_file.assert_not_called()

    def test_skips_books_matched_by_title(self, temp_dir):
        client = _client([_drive_file("Atomic Habits - James Clear.pdf", "new-hash")])
        ingest = MagicMock()
        result = sync_drive_folder(client, [_book("Atomic Habits", "old-hash")], ingest, temp_dir)
        assert result.already_ingested == 1
        ingest.assert_not_called()

    def test_ingests_new_books_with_drive_file_id(self, temp_dir):
        client = _client([_drive_file("Deep Work.epub", "h1", "application/epub+zip")])
        ingest = MagicMock(side_effect=_ingest_ok)
        result = sync_drive_folder(client, [_book("Atomic Habits", "abc")], ingest, temp_dir)
        assert [b["title"] for b in result.ingested] == ["Deep Work"]
        kwargs = ingest.call_args.kwargs
        assert kwargs["path"] == temp_dir / "Deep Work.epub"
        assert kwargs["drive_file_id"] == "id-Deep Work.epub"

    def test_caps_books_per_run(self, temp_dir):
        client = _client([_drive_file(f"Book {n}.pdf", f"h{n}") for n in "ABC"])
        result = sync_drive_folder(client, [], MagicMock(side_effect=_ingest_ok), temp_dir, max_books=2)
        assert len(result.ingested) == 2
        assert result.pending == ["Book C.pdf"]

    def test_one_failure_does_not_stop_the_rest(self, temp_dir):
        client = _client([_drive_file("Bad.pdf", "h1"), _drive_file("Good.pdf", "h2")])

        def ingest(path: Path, **kw):
            if path.stem == "Bad":
                raise RuntimeError("parse error")
            return _ingest_ok(path)

        result = sync_drive_folder(client, [], ingest, temp_dir)
        assert result.failed == [("Bad.pdf", "parse error")]
        assert [b["title"] for b in result.ingested] == ["Good"]

    def test_duplicate_found_at_ingest_counts_as_already_ingested(self, temp_dir):
        client = _client([_drive_file("Copy.pdf", None)])
        result = sync_drive_folder(client, [], MagicMock(return_value=None), temp_dir)
        assert result.already_ingested == 1
        assert result.ingested == []

    def test_dry_run_lists_without_downloading(self, temp_dir):
        client = _client([_drive_file("New.pdf", "h1")])
        ingest = MagicMock()
        result = sync_drive_folder(client, [], ingest, temp_dir, dry_run=True)
        assert result.pending == ["New.pdf"]
        ingest.assert_not_called()
        client.download_file.assert_not_called()


class TestLocalFilename:
    def test_adds_extension_from_mime_type(self):
        assert local_filename(_drive_file("Book", "h", "application/epub+zip")) == "Book.epub"

    def test_keeps_existing_extension(self):
        assert local_filename(_drive_file("Book.pdf", "h")) == "Book.pdf"


def test_listing_requests_md5_checksum():
    client = GoogleDriveClient()
    service = MagicMock()
    service.files.return_value.list.return_value.execute.return_value = {
        "files": [{"id": "1", "name": "A.pdf", "mimeType": "application/pdf", "md5Checksum": "abc"}]
    }
    client._service = service
    files = client._list_book_files("folder")
    assert files[0].md5_checksum == "abc"
    assert "md5Checksum" in service.files.return_value.list.call_args.kwargs["fields"]


class TestAddedSinceCutoff:
    def test_skips_files_added_before_cutoff(self, temp_dir):
        from datetime import datetime, timezone

        old = _drive_file("Old.pdf", "h1")
        old.created_time = "2026-09-08T06:11:37.336Z"
        new = _drive_file("New.pdf", "h2")
        new.created_time = "2026-10-30T01:00:00.000Z"
        client = _client([old, new])
        cutoff = datetime(2026, 10, 29, 0, 30, tzinfo=timezone.utc)
        result = sync_drive_folder(
            client, [], MagicMock(side_effect=_ingest_ok), temp_dir, added_since=cutoff
        )
        assert result.older_than_cutoff == 1
        assert [b["title"] for b in result.ingested] == ["New"]

    def test_listing_requests_created_time(self):
        client = GoogleDriveClient()
        service = MagicMock()
        service.files.return_value.list.return_value.execute.return_value = {
            "files": [{"id": "1", "name": "A.pdf", "mimeType": "application/pdf",
                       "createdTime": "2026-10-30T01:00:00Z"}]
        }
        client._service = service
        assert client._list_book_files("folder")[0].created_time == "2026-10-30T01:00:00Z"


class TestReingestBooks:
    def test_rebuilds_each_book_in_place(self, temp_dir):
        from book_companion.drive_sync import reingest_books

        books = [_book("Atomic Habits", "abc"), _book("Situated", "old")]
        client = _client([
            _drive_file("Renamed copy.pdf", "abc"),  # checksum match
            _drive_file("Situated - Mind in World.epub", "new", "application/epub+zip"),  # title match
            _drive_file("Unrelated.pdf", "zzz"),
        ])
        ingest = MagicMock(side_effect=_ingest_ok)
        result = reingest_books(client, books, ingest, temp_dir)

        assert len(result.reingested) == 2 and not result.missing and not result.failed
        replaced = {c.kwargs["replace_book_id"]: c.kwargs["path"].name for c in ingest.call_args_list}
        assert replaced == {books[0].id: "Renamed copy.pdf", books[1].id: "Situated - Mind in World.epub"}

    def test_reports_books_missing_from_drive(self, temp_dir):
        from book_companion.drive_sync import reingest_books

        book = _book("Gone", "x")
        result = reingest_books(_client([]), [book], MagicMock(), temp_dir)
        assert result.missing == [f"Gone ({book.id})"]

    def test_dry_run_plans_without_downloading(self, temp_dir):
        from book_companion.drive_sync import reingest_books

        client = _client([_drive_file("a.pdf", "abc")])
        result = reingest_books(client, [_book("Atomic Habits", "abc")], MagicMock(), temp_dir, dry_run=True)
        assert [f for _, f in result.planned] == ["a.pdf"]
        client.download_file.assert_not_called()
