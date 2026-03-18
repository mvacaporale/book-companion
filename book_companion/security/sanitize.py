"""Filename and path sanitization utilities."""

import re
from pathlib import Path


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal and other attacks.

    Removes directory components, path traversal attempts (..),
    and characters that are unsafe in filenames across platforms.

    Args:
        filename: The raw filename (potentially from untrusted source like Google Drive)

    Returns:
        A safe filename with no path components or unsafe characters.
        Returns 'downloaded_file' if the result would be empty.

    Examples:
        >>> sanitize_filename("../../../etc/passwd")
        'passwd'
        >>> sanitize_filename("My Book.pdf")
        'My Book.pdf'
        >>> sanitize_filename("../malicious/../../../file.txt")
        'file.txt'
        >>> sanitize_filename("file:name<with>bad|chars?.pdf")
        'file_name_with_bad_chars_.pdf'
    """
    if not filename:
        return "downloaded_file"

    # Normalize Windows backslashes to forward slashes before processing
    # This ensures consistent behavior across platforms
    filename = filename.replace("\\", "/")

    # Extract just the filename, removing any directory components
    name = Path(filename).name

    # Remove any remaining path traversal attempts (in case of edge cases)
    # Replace .. sequences
    name = re.sub(r"\.{2,}", ".", name)

    # Remove or replace characters that are unsafe in filenames
    # Windows: < > : " / \ | ? *
    # Unix: / and null
    # Also remove control characters
    unsafe_chars = r'[<>:"/\\|?*\x00-\x1f]'
    name = re.sub(unsafe_chars, "_", name)

    # Don't allow hidden files (starting with .)
    name = name.lstrip(".")

    # Collapse multiple underscores/spaces
    name = re.sub(r"_+", "_", name)
    name = re.sub(r"\s+", " ", name)

    # Trim whitespace and underscores from ends
    name = name.strip(" _")

    # Ensure we have a valid filename
    if not name:
        return "downloaded_file"

    # Limit length (most filesystems support 255 bytes)
    max_length = 200  # Leave room for path
    if len(name) > max_length:
        # Try to preserve extension
        ext = Path(name).suffix
        if ext and len(ext) < 10:
            name = name[: max_length - len(ext)] + ext
        else:
            name = name[:max_length]

    return name
