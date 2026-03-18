"""Pytest fixtures for book-companion tests."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_bookrc_dir(temp_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a mock ~/.bookrc directory for testing."""
    bookrc_dir = temp_dir / ".bookrc"
    bookrc_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("BOOKRC_DB_PATH", str(bookrc_dir))
    return bookrc_dir


@pytest.fixture
def sample_book_text() -> str:
    """Sample book text for testing."""
    return """
    # Chapter 1: Introduction

    This is the introduction to our sample book. It contains important concepts
    that will be explored throughout the text.

    ## Key Points

    - Point one about the subject
    - Point two with more details
    - Point three for completeness

    # Chapter 2: Main Content

    The main content of the book dives deeper into the subject matter.
    We explore various aspects and provide examples to illustrate concepts.

    ## Case Study: Example Corp

    Example Corp implemented these principles with great success.
    Their journey shows how theory translates to practice.

    # Chapter 3: Conclusion

    In conclusion, we've covered the main points and provided actionable advice.
    """


@pytest.fixture
def sample_markdown_file(temp_dir: Path, sample_book_text: str) -> Path:
    """Create a sample markdown file for testing."""
    file_path = temp_dir / "sample_book.md"
    file_path.write_text(sample_book_text)
    return file_path


@pytest.fixture(autouse=True)
def reset_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset environment variables that might affect tests."""
    # Clear any OAuth-related env vars
    for var in [
        "MCP_OAUTH_ENABLED",
        "MCP_DEBUG",
        "GOOGLE_DRIVE_TOKEN",
        "GOOGLE_DRIVE_TOKEN_B64",
    ]:
        monkeypatch.delenv(var, raising=False)
