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

# Environment variable names for Cloud Run deployment
ENV_GOOGLE_DRIVE_TOKEN = "GOOGLE_DRIVE_TOKEN"
ENV_GOOGLE_DRIVE_TOKEN_B64 = "GOOGLE_DRIVE_TOKEN_B64"
ENV_GOOGLE_DRIVE_FOLDER_ID = "GOOGLE_DRIVE_FOLDER_ID"


def _get_credentials_from_env() -> Optional[Credentials]:
    """
    Try to load credentials from environment variable.

    For Cloud Run deployment, set either:
    - GOOGLE_DRIVE_TOKEN: Raw JSON contents of token file
    - GOOGLE_DRIVE_TOKEN_B64: Base64-encoded JSON (preferred, avoids shell escaping)

    Returns:
        Credentials if env var is set and valid, None otherwise
    """
    import base64

    # Try base64-encoded token first (preferred for Cloud Run)
    token_b64 = os.environ.get(ENV_GOOGLE_DRIVE_TOKEN_B64)
    if token_b64:
        try:
            token_json = base64.b64decode(token_b64).decode("utf-8")
            token_data = json.loads(token_json)
            creds = Credentials.from_authorized_user_info(token_data, SCOPES)

            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())

            if creds and creds.valid:
                return creds
        except Exception:
            pass

    # Fall back to raw JSON token
    token_json = os.environ.get(ENV_GOOGLE_DRIVE_TOKEN)
    if not token_json:
        return None

    try:
        token_data = json.loads(token_json)
        creds = Credentials.from_authorized_user_info(token_data, SCOPES)

        # Refresh if expired
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        if creds and creds.valid:
            return creds
    except Exception:
        pass

    return None


def get_credentials(
    credentials_path: Optional[Path] = None,
    token_path: Optional[Path] = None,
) -> Optional[Credentials]:
    """
    Get valid Google API credentials.

    Checks in order:
    1. GOOGLE_DRIVE_TOKEN environment variable (for Cloud Run)
    2. Token file on disk (for local development)

    Args:
        credentials_path: Path to OAuth client credentials JSON
        token_path: Path to stored token JSON

    Returns:
        Valid Credentials object, or None if not available
    """
    # First, try environment variable (Cloud Run)
    creds = _get_credentials_from_env()
    if creds:
        return creds

    # Fall back to file-based credentials (local)
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
    """Save credentials to token file with secure permissions."""
    token_path.parent.mkdir(parents=True, exist_ok=True)
    with open(token_path, "w") as f:
        f.write(creds.to_json())
    # Set restrictive permissions (owner read/write only)
    os.chmod(token_path, 0o600)


def get_config() -> dict:
    """
    Load Google Drive configuration.

    Checks in order:
    1. Environment variables (for Cloud Run):
       - GOOGLE_DRIVE_FOLDER_ID
    2. Config file on disk (for local development)

    Returns:
        Configuration dict with keys like 'default_folder_id', 'cache_ttl_seconds'
    """
    config = {}

    # Load from config file first
    config_path = Path.home() / ".bookrc" / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                file_config = json.load(f)
            config = file_config.get("google_drive", {})
        except Exception:
            pass

    # Override with environment variables (Cloud Run)
    folder_id = os.environ.get(ENV_GOOGLE_DRIVE_FOLDER_ID)
    if folder_id:
        config["default_folder_id"] = folder_id

    return config


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
