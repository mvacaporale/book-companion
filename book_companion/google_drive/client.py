"""Google Drive API client wrapper for book operations."""

import io
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from rapidfuzz import fuzz

from .auth import authenticate, get_config, DEFAULT_CREDENTIALS_PATH, DEFAULT_TOKEN_PATH


@dataclass
class DriveFile:
    """Represents a file in Google Drive."""

    id: str
    name: str
    mime_type: str
    size: Optional[int] = None
    modified_time: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "mime_type": self.mime_type,
            "size": self.size,
            "modified_time": self.modified_time,
        }


class GoogleDriveClient:
    """Wrapper for Google Drive API operations focused on book files."""

    # Supported book file types
    BOOK_MIMETYPES = [
        "application/pdf",
        "application/epub+zip",
        "text/markdown",
        "text/plain",
    ]

    # File extensions to include in searches
    BOOK_EXTENSIONS = [".pdf", ".epub", ".md", ".markdown", ".txt"]

    def __init__(
        self,
        credentials_path: Optional[Path] = None,
        token_path: Optional[Path] = None,
    ):
        """
        Initialize Google Drive client.

        Args:
            credentials_path: Path to OAuth client credentials JSON
            token_path: Path to stored token JSON
        """
        self.credentials_path = credentials_path or DEFAULT_CREDENTIALS_PATH
        self.token_path = token_path or DEFAULT_TOKEN_PATH
        self._service = None
        self._file_cache: dict[str, tuple[list[DriveFile], float]] = {}
        self._cache_ttl = get_config().get("cache_ttl_seconds", 300)

    @property
    def service(self):
        """Lazy initialization of Drive service."""
        if self._service is None:
            creds = authenticate(self.credentials_path, self.token_path)
            self._service = build("drive", "v3", credentials=creds)
        return self._service

    def search_books(
        self,
        query: str,
        folder_id: Optional[str] = None,
        threshold: int = 60,
        max_results: int = 5,
    ) -> list[tuple[DriveFile, int]]:
        """
        Search for books matching query with fuzzy matching.

        Uses cached file listing when available. Performs fuzzy matching
        on filenames to handle variations in book titles.

        Args:
            query: Book title or keywords to search for
            folder_id: Optional Drive folder ID to search in
            threshold: Minimum fuzzy match score (0-100)
            max_results: Maximum number of results to return

        Returns:
            List of (DriveFile, score) tuples, sorted by score descending
        """
        # Get folder from config if not specified
        if folder_id is None:
            folder_id = get_config().get("default_folder_id")

        # List all book files (cached)
        files = self._list_book_files(folder_id)

        # Fuzzy match against query
        results = []
        for file in files:
            clean_name = self._clean_book_filename(file.name)
            score = fuzz.token_set_ratio(query.lower(), clean_name.lower())
            if score >= threshold:
                results.append((file, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:max_results]

    def download_file(self, file_id: str, dest_path: Path) -> Path:
        """
        Download a file from Drive to local path.

        Args:
            file_id: Google Drive file ID
            dest_path: Destination path for downloaded file

        Returns:
            Path to downloaded file

        Raises:
            Exception: If download fails
        """
        # Ensure parent directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Create download request
        request = self.service.files().get_media(fileId=file_id)

        # Download to file
        with open(dest_path, "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        return dest_path

    def get_file_metadata(self, file_id: str) -> DriveFile:
        """
        Get metadata for a file.

        Args:
            file_id: Google Drive file ID

        Returns:
            DriveFile with metadata

        Raises:
            Exception: If file not found or API error
        """
        result = (
            self.service.files()
            .get(fileId=file_id, fields="id,name,mimeType,size,modifiedTime")
            .execute()
        )

        return DriveFile(
            id=result["id"],
            name=result["name"],
            mime_type=result["mimeType"],
            size=int(result.get("size", 0)) if result.get("size") else None,
            modified_time=result.get("modifiedTime"),
        )

    def _list_book_files(self, folder_id: Optional[str] = None) -> list[DriveFile]:
        """
        List all book files in folder (with caching).

        Args:
            folder_id: Optional folder ID to list from

        Returns:
            List of DriveFile objects
        """
        cache_key = folder_id or "__root__"
        current_time = time.time()

        # Check cache
        if cache_key in self._file_cache:
            files, cached_at = self._file_cache[cache_key]
            if current_time - cached_at < self._cache_ttl:
                return files

        # Build query for book files
        query_parts = []

        # Filter by folder if specified
        if folder_id:
            query_parts.append(f"'{folder_id}' in parents")

        # Filter by mime types
        mime_filters = " or ".join(
            f"mimeType='{mt}'" for mt in self.BOOK_MIMETYPES
        )
        query_parts.append(f"({mime_filters})")

        # Exclude trashed files
        query_parts.append("trashed=false")

        query = " and ".join(query_parts)

        # Fetch all pages
        files = []
        page_token = None

        while True:
            results = (
                self.service.files()
                .list(
                    q=query,
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
                    pageToken=page_token,
                    pageSize=100,
                )
                .execute()
            )

            for item in results.get("files", []):
                # Additional extension check for text/plain files
                name = item["name"]
                if item["mimeType"] == "text/plain":
                    if not any(name.lower().endswith(ext) for ext in self.BOOK_EXTENSIONS):
                        continue

                files.append(
                    DriveFile(
                        id=item["id"],
                        name=item["name"],
                        mime_type=item["mimeType"],
                        size=int(item.get("size", 0)) if item.get("size") else None,
                        modified_time=item.get("modifiedTime"),
                    )
                )

            page_token = results.get("nextPageToken")
            if not page_token:
                break

        # Cache results
        self._file_cache[cache_key] = (files, current_time)

        return files

    def _clean_book_filename(self, filename: str) -> str:
        """
        Clean filename for fuzzy matching.

        Removes extension, common suffixes, and normalizes spacing.

        Args:
            filename: Original filename

        Returns:
            Cleaned filename for matching
        """
        # Remove extension
        name = Path(filename).stem

        # Remove common patterns
        patterns_to_remove = [
            r"\s*-\s*[A-Z][a-z]+\s+[A-Z][a-z]+$",  # " - Author Name"
            r"\s*\([^)]+\)$",  # " (Publisher)" etc.
            r"\s*\[[^\]]+\]$",  # " [Format]" etc.
            r"\s*_+\s*",  # Underscores to spaces
        ]

        for pattern in patterns_to_remove:
            name = re.sub(pattern, " ", name)

        # Normalize whitespace
        name = " ".join(name.split())

        return name

    def clear_cache(self) -> None:
        """Clear the file listing cache."""
        self._file_cache.clear()


def get_drive_client() -> GoogleDriveClient:
    """
    Get a configured GoogleDriveClient instance.

    Returns:
        Configured client using default credential paths
    """
    return GoogleDriveClient()
