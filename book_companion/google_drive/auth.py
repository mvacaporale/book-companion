"""OAuth 2.0 authentication for Google Drive API."""

import json
import os
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Read-only scope for Drive access
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Default paths for credentials
DEFAULT_CREDENTIALS_PATH = Path.home() / ".bookrc" / "google_credentials.json"
DEFAULT_TOKEN_PATH = Path.home() / ".bookrc" / "google_token.json"


def get_credentials(
    credentials_path: Optional[Path] = None,
    token_path: Optional[Path] = None,
) -> Optional[Credentials]:
    """
    Get valid Google API credentials.

    Attempts to load existing credentials from token file.
    If credentials are expired, attempts to refresh them.
    Returns None if no valid credentials are available.

    Args:
        credentials_path: Path to OAuth client credentials JSON
        token_path: Path to stored token JSON

    Returns:
        Valid Credentials object, or None if not available
    """
    token_path = token_path or DEFAULT_TOKEN_PATH
    credentials_path = credentials_path or DEFAULT_CREDENTIALS_PATH

    creds = None

    # Load existing token if available
    if token_path.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        except Exception:
            # Token file is invalid, will need to re-authenticate
            pass

    # Check if credentials are valid
    if creds and creds.valid:
        return creds

    # Try to refresh expired credentials
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save_credentials(creds, token_path)
            return creds
        except Exception:
            # Refresh failed, will need to re-authenticate
            pass

    return None


def authenticate(
    credentials_path: Optional[Path] = None,
    token_path: Optional[Path] = None,
) -> Credentials:
    """
    Authenticate with Google Drive API.

    Uses existing credentials if available and valid.
    Otherwise, initiates OAuth flow requiring user interaction.

    Args:
        credentials_path: Path to OAuth client credentials JSON
        token_path: Path to store/load token JSON

    Returns:
        Valid Credentials object

    Raises:
        FileNotFoundError: If credentials file doesn't exist
        ValueError: If authentication fails
    """
    token_path = token_path or DEFAULT_TOKEN_PATH
    credentials_path = credentials_path or DEFAULT_CREDENTIALS_PATH

    # Try to get existing credentials
    creds = get_credentials(credentials_path, token_path)
    if creds:
        return creds

    # Need to run OAuth flow
    if not credentials_path.exists():
        raise FileNotFoundError(
            f"OAuth credentials not found at {credentials_path}. "
            "Download credentials from Google Cloud Console and save to this path."
        )

    # Run OAuth flow
    flow = InstalledAppFlow.from_client_secrets_file(
        str(credentials_path),
        SCOPES,
    )
    creds = flow.run_local_server(port=0)

    # Save credentials for future use
    _save_credentials(creds, token_path)

    return creds


def setup_drive_auth(
    credentials_path: Optional[Path] = None,
    token_path: Optional[Path] = None,
) -> bool:
    """
    Interactive setup for Google Drive authentication.

    Guides user through OAuth flow and saves refresh token.
    Called by CLI 'setup-drive' command.

    Args:
        credentials_path: Path to OAuth client credentials JSON
        token_path: Path to store token JSON

    Returns:
        True if authentication successful, False otherwise
    """
    token_path = token_path or DEFAULT_TOKEN_PATH
    credentials_path = credentials_path or DEFAULT_CREDENTIALS_PATH

    # Check for existing valid credentials
    existing_creds = get_credentials(credentials_path, token_path)
    if existing_creds:
        return True

    # Check for credentials file
    if not credentials_path.exists():
        raise FileNotFoundError(
            f"OAuth credentials file not found at: {credentials_path}\n\n"
            "To set up Google Drive access:\n"
            "1. Go to https://console.cloud.google.com/apis/credentials\n"
            "2. Create OAuth 2.0 Client ID (Desktop app type)\n"
            "3. Download the JSON file\n"
            f"4. Save it to: {credentials_path}"
        )

    # Ensure parent directory exists
    token_path.parent.mkdir(parents=True, exist_ok=True)

    # Run OAuth flow
    try:
        creds = authenticate(credentials_path, token_path)
        return creds is not None and creds.valid
    except Exception as e:
        raise ValueError(f"Authentication failed: {e}")


def is_authenticated(
    credentials_path: Optional[Path] = None,
    token_path: Optional[Path] = None,
) -> bool:
    """
    Check if valid Google Drive credentials are available.

    Args:
        credentials_path: Path to OAuth client credentials JSON
        token_path: Path to token JSON

    Returns:
        True if valid credentials exist, False otherwise
    """
    creds = get_credentials(credentials_path, token_path)
    return creds is not None and creds.valid


def _save_credentials(creds: Credentials, token_path: Path) -> None:
    """Save credentials to token file."""
    token_path.parent.mkdir(parents=True, exist_ok=True)
    with open(token_path, "w") as f:
        f.write(creds.to_json())


def get_config() -> dict:
    """
    Load Google Drive configuration from config file.

    Returns:
        Configuration dict with keys like 'default_folder_id', 'cache_ttl_seconds'
    """
    config_path = Path.home() / ".bookrc" / "config.json"

    if not config_path.exists():
        return {}

    try:
        with open(config_path) as f:
            config = json.load(f)
        return config.get("google_drive", {})
    except Exception:
        return {}


def save_config(drive_config: dict) -> None:
    """
    Save Google Drive configuration to config file.

    Args:
        drive_config: Dict with keys like 'default_folder_id', 'cache_ttl_seconds'
    """
    config_path = Path.home() / ".bookrc" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config
    full_config = {}
    if config_path.exists():
        try:
            with open(config_path) as f:
                full_config = json.load(f)
        except Exception:
            pass

    # Update google_drive section
    full_config["google_drive"] = drive_config

    with open(config_path, "w") as f:
        json.dump(full_config, f, indent=2)
