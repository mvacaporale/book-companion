"""Tests for filename sanitization."""

import pytest

from book_companion.security.sanitize import sanitize_filename


class TestSanitizeFilename:
    """Test cases for sanitize_filename function."""

    def test_normal_filename(self) -> None:
        """Normal filenames should pass through unchanged."""
        assert sanitize_filename("My Book.pdf") == "My Book.pdf"
        assert sanitize_filename("document.epub") == "document.epub"
        assert sanitize_filename("notes.txt") == "notes.txt"

    def test_path_traversal_unix(self) -> None:
        """Path traversal attempts on Unix should be blocked."""
        assert sanitize_filename("../../../etc/passwd") == "passwd"
        assert sanitize_filename("../../secret.txt") == "secret.txt"
        assert sanitize_filename("./hidden/file.txt") == "file.txt"

    def test_path_traversal_windows(self) -> None:
        """Path traversal attempts on Windows should be blocked."""
        assert sanitize_filename("..\\..\\Windows\\System32\\config") == "config"
        assert sanitize_filename("C:\\Users\\secret.txt") == "secret.txt"

    def test_absolute_paths(self) -> None:
        """Absolute paths should be reduced to filename only."""
        assert sanitize_filename("/etc/passwd") == "passwd"
        assert sanitize_filename("/home/user/documents/book.pdf") == "book.pdf"
        assert sanitize_filename("C:\\Users\\user\\book.pdf") == "book.pdf"

    def test_hidden_files(self) -> None:
        """Hidden files (starting with .) should have dot stripped."""
        assert sanitize_filename(".hidden") == "hidden"
        assert sanitize_filename(".bashrc") == "bashrc"
        assert sanitize_filename("..double") == "double"

    def test_unsafe_characters(self) -> None:
        """Unsafe characters should be replaced with underscore."""
        assert sanitize_filename("file:name.pdf") == "file_name.pdf"
        assert sanitize_filename("file<name>.pdf") == "file_name_.pdf"
        assert sanitize_filename('file"name".pdf') == "file_name_.pdf"
        assert sanitize_filename("file|name.pdf") == "file_name.pdf"
        assert sanitize_filename("file?name.pdf") == "file_name.pdf"
        assert sanitize_filename("file*name.pdf") == "file_name.pdf"

    def test_empty_filename(self) -> None:
        """Empty or whitespace-only filenames should return default."""
        assert sanitize_filename("") == "downloaded_file"
        assert sanitize_filename("   ") == "downloaded_file"
        assert sanitize_filename("...") == "downloaded_file"

    def test_multiple_dots(self) -> None:
        """Multiple consecutive dots should be collapsed."""
        assert sanitize_filename("file....txt") == "file.txt"
        assert sanitize_filename("...file...txt...") == "file.txt."

    def test_long_filename(self) -> None:
        """Long filenames should be truncated while preserving extension."""
        long_name = "a" * 300 + ".pdf"
        result = sanitize_filename(long_name)
        assert len(result) <= 200
        assert result.endswith(".pdf")

    def test_unicode_filenames(self) -> None:
        """Unicode characters in filenames should be preserved."""
        assert sanitize_filename("日本語ファイル.pdf") == "日本語ファイル.pdf"
        assert sanitize_filename("émojis_📚.epub") == "émojis_📚.epub"

    def test_whitespace_handling(self) -> None:
        """Multiple spaces should be collapsed."""
        assert sanitize_filename("file   name.pdf") == "file name.pdf"
        assert sanitize_filename("  leading.pdf") == "leading.pdf"
        assert sanitize_filename("trailing.pdf  ") == "trailing.pdf"

    def test_null_bytes(self) -> None:
        """Null bytes and control characters should be removed."""
        assert sanitize_filename("file\x00name.pdf") == "file_name.pdf"
        assert sanitize_filename("file\x1fname.pdf") == "file_name.pdf"

    def test_complex_attack_patterns(self) -> None:
        """Complex attack patterns should be sanitized."""
        # Combination of path traversal and unsafe chars
        assert sanitize_filename("../../../<script>alert.html") == "script_alert.html"
        # URL-like path traversal
        assert sanitize_filename("..%2F..%2Fetc/passwd") == "passwd"
        # Nested traversal
        assert sanitize_filename("....//....//file.txt") == "file.txt"
