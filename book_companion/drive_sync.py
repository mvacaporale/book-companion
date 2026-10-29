"""Automatic ingestion of new books from the configured Google Drive folder.

`sync_drive_folder` lists the books folder, skips files that are already
ingested, and downloads + ingests the rest. It runs from `bookrc drive sync`
(the `drive-sync.yml` workflow, started by the Drive watcher in `drive-watcher/`).

A Drive file counts as already ingested when either:
  - its Drive md5Checksum equals a registered book's file_hash (both are the
    MD5 of the file bytes), so the common case needs no download; or
  - its cleaned filename fuzzy-matches a registered title (>= 80, the same
    rule `find_book_in_drive` uses for `is_ingested`), which covers books
    ingested from a different copy of the file before this job existed.

Files added to Drive before `added_since` (Drive createdTime) are skipped, so
the backlog that predates this job isn't ingested; the workflow sets it via
DRIVE_SYNC_SINCE. Note a file moved into the folder keeps its original
createdTime, so an old file moved in is skipped too (ingest it by hand).

Failures are per-book and non-fatal; the caller decides the exit code.
"""

import logging
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from rapidfuzz import fuzz

from book_companion.google_drive.client import DriveFile, GoogleDriveClient
from book_companion.models import Book
from book_companion.security import sanitize_filename

logger = logging.getLogger(__name__)

TITLE_MATCH_THRESHOLD = 80

MIME_TO_EXT = {
    "application/pdf": ".pdf",
    "application/epub+zip": ".epub",
    "text/markdown": ".md",
    "text/plain": ".md",
}
RECOGNIZED_EXTENSIONS = {".pdf", ".epub", ".md", ".txt", ".markdown"}


@dataclass
class SyncResult:
    """What one sync pass did."""

    seen: int = 0
    already_ingested: int = 0
    older_than_cutoff: int = 0
    ingested: list[dict] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)  # new files left for a later run (or dry run)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (file name, error)


def local_filename(file: DriveFile) -> str:
    """Safe local filename with an extension the parsers recognize."""
    name = sanitize_filename(file.name)
    if Path(name).suffix.lower() not in RECOGNIZED_EXTENSIONS:
        name += MIME_TO_EXT.get(file.mime_type, "")
    return name


def is_already_ingested(
    client: GoogleDriveClient, file: DriveFile, books: list[Book]
) -> bool:
    """True when `file` matches a registered book by checksum or title."""
    if file.md5_checksum and any(b.file_hash == file.md5_checksum for b in books):
        return True
    clean_name = client._clean_book_filename(file.name).lower()
    return any(
        fuzz.token_set_ratio(clean_name, b.title.lower()) >= TITLE_MATCH_THRESHOLD
        for b in books
    )


def sync_drive_folder(
    client: GoogleDriveClient,
    books: list[Book],
    ingest_fn: Callable[..., Optional[dict]],
    downloads_dir: Path,
    folder_id: Optional[str] = None,
    max_books: int = 5,
    dry_run: bool = False,
    skip_summary: bool = False,
    added_since: Optional[datetime] = None,
) -> SyncResult:
    """Ingest Drive books that aren't registered yet, at most `max_books` per run."""
    result = SyncResult()
    files = client._list_book_files(folder_id)
    result.seen = len(files)

    if added_since:
        recent = [f for f in files if not _created_before(f, added_since)]
        result.older_than_cutoff = len(files) - len(recent)
        files = recent

    new_files = [f for f in files if not is_already_ingested(client, f, books)]
    result.already_ingested = len(files) - len(new_files)

    if dry_run:
        result.pending = [f.name for f in new_files]
        return result

    for file in new_files[:max_books]:
        try:
            dest = client.download_file(file.id, downloads_dir / local_filename(file))
            ingested = ingest_fn(
                path=dest, drive_file_id=file.id, skip_summary=skip_summary
            )
            if ingested is None:
                # ingest_book's own hash check found a duplicate.
                result.already_ingested += 1
            else:
                result.ingested.append(ingested)
        except Exception as e:
            logger.exception("Drive sync failed for %s", file.name)
            result.failed.append((file.name, str(e)))

    result.pending = [f.name for f in new_files[max_books:]]
    return result


def _created_before(file: DriveFile, cutoff: datetime) -> bool:
    if not file.created_time:
        return False
    return datetime.fromisoformat(file.created_time.replace("Z", "+00:00")) < cutoff
