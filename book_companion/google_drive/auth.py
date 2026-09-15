"""Authentication for Google Drive API.

Two supported credential sources, tried in order:
1. Service account via Application Default Credentials (Cloud Run runtime
   identity, or GOOGLE_APPLICATION_CREDENTIALS locally). Users grant access
   by sharing their books folder with the service account's email.
2. Local user OAuth token from `bookrc setup-drive` (local development).
"""

import json
import os
from pathlib import Path
from typing import Optional

import google.auth
from google.auth.exceptions import DefaultCredentialsError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Read-only scope for Drive access
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Default paths for credentials
DEFAULT_CREDENTIALS_PATH = Path.home() / ".bookrc" / "google_credentials.json"
DEFAULT_TOKEN_PATH = Path.home() / ".bookrc" / "google_token.json"

# Reuse the Google Workspace CLI (`gws`) OAuth token for Drive access. gws is
# authenticated as the user with a long-lived (production-app) refresh token, so
# reusing it avoids book-companion's own OAuth flow and its 7-day token expiry.
ENV_GWS_CREDENTIALS_PATH = "BOOKRC_GWS_CREDENTIALS"
DEFAULT_GWS_CREDENTIALS_PATH = Path.home() / ".config" / "gws" / "credentials.json"
DEFAULT_GWS_CLIENT_SECRET_PATH = Path.home() / ".config" / "gws" / "client_secret.json"

# Environment variable names for Cloud Run deployment
ENV_GOOGLE_DRIVE_FOLDER_ID = "GOOGLE_DRIVE_FOLDER_ID"


def _get_service_account_credentials():
    """
    Try to load service account credentials via Application Default Credentials.

    On Cloud Run this resolves to the service's runtime identity; locally it
    resolves to GOOGLE_APPLICATION_CREDENTIALS if set. End-user ADC credentials
    (e.g. from `gcloud auth application-default login`) are ignored — they
    don't carry Drive scopes.

    Returns:
        Scoped service account credentials, or None if unavailable
    """
    try:
        creds, _ = google.auth.default(scopes=SCOPES)
    except DefaultCredentialsError:
        return None

    # Only accept service-account-style identities (JSON key or metadata
    # server). User ADC credentials can't be scoped for Drive.
    if not hasattr(creds, "service_account_email"):
        return None

    return creds


def _unwrap_oauth_client(data: dict) -> dict:
    """Flatten an 'installed'/'web' OAuth-client wrapper into the top level."""
    for key in ("installed", "web"):
        wrapper = data.get(key)
        if isinstance(wrapper, dict):
            merged = dict(wrapper)
            merged.update({k: v for k, v in data.items() if k != key})
            return merged
    return data


def _get_gws_credentials():
    """
    Reuse the Google Workspace CLI (`gws`) OAuth token for Drive access.

    gws holds a long-lived refresh token (its OAuth app is published, so the
    token doesn't hit Google's 7-day testing-mode expiry). We mint a Drive access
    token from that refresh token without creating a new token or mutating gws's
    files, and cache the access token to DEFAULT_TOKEN_PATH.

    Returns:
        Valid credentials, or None if gws credentials are absent or unusable.
    """
    gws_path = Path(os.getenv(ENV_GWS_CREDENTIALS_PATH, str(DEFAULT_GWS_CREDENTIALS_PATH)))
    if not gws_path.exists():
        return None

    try:
        with open(gws_path) as f:
            data = _unwrap_oauth_client(json.load(f))
    except Exception:
        return None

    client_id = data.get("client_id")
    client_secret = data.get("client_secret")
    refresh_token = data.get("refresh_token")

    # gws may split the OAuth client into a sibling client_secret.json.
    if not (client_id and client_secret) and DEFAULT_GWS_CLIENT_SECRET_PATH.exists():
        try:
            with open(DEFAULT_GWS_CLIENT_SECRET_PATH) as f:
                client = _unwrap_oauth_client(json.load(f))
            client_id = client_id or client.get("client_id")
            client_secret = client_secret or client.get("client_secret")
        except Exception:
            pass

    if not (refresh_token and client_id and client_secret):
        return None

    creds = Credentials(
        token=data.get("token") or data.get("access_token"),
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
        token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
        scopes=["https://www.googleapis.com/auth/drive"],
    )

    try:
        creds.refresh(Request())
    except Exception:
        return None

    # Cache the access token for reuse; never write back to the gws file.
    try:
        _save_credentials(creds, DEFAULT_TOKEN_PATH)
    except Exception:
        pass

    return creds


def get_service_account_email() -> Optional[str]:
    """
    Return the service account email in use, or None if not using one.

    This is the address users share their books folder with.
    """
    creds = _get_service_account_credentials()
    if creds is None:
        return None

    email = getattr(creds, "service_account_email", None)
    # Metadata-server credentials report "default" until a token is fetched
    if email == "default":
        try:
            creds.refresh(Request())
            email = getattr(creds, "service_account_email", None)
        except Exception:
            pass
    return email


def get_credentials(
    credentials_path: Optional[Path] = None,
    token_path: Optional[Path] = None,
):
    """
    Get valid Google API credentials.

    Checks in order:
    1. Service account via Application Default Credentials (Cloud Run)
    2. Token file on disk (local development)

    Args:
        credentials_path: Path to OAuth client credentials JSON
        token_path: Path to stored token JSON

    Returns:
        Valid credentials object, or None if not available
    """
    # First, try service account credentials (Cloud Run)
    creds = _get_service_account_credentials()
    if creds:
        return creds

    # Next, reuse the gws (Google Workspace CLI) OAuth token if available
    creds = _get_gws_credentials()
    if creds and creds.valid:
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
):
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


def verify_drive_access(folder_id: Optional[str] = None) -> Optional[str]:
    """
    Verify Drive credentials work and the books folder is accessible.

    Uses files.get on the folder rather than files.list because listing an
    unshared folder silently returns no results instead of an error.

    Args:
        folder_id: Drive folder ID to check; defaults to configured folder

    Returns:
        None if access works, otherwise an actionable error message
    """
    if folder_id is None:
        folder_id = get_config().get("default_folder_id")

    creds = get_credentials()
    if creds is None:
        return (
            "Google Drive not configured. "
            "Run 'bookrc setup-drive' locally, or deploy with a service account."
        )

    if not folder_id:
        return (
            "No Drive folder configured. "
            f"Set {ENV_GOOGLE_DRIVE_FOLDER_ID} or 'default_folder_id' in config."
        )

    try:
        from googleapiclient.discovery import build

        service = build("drive", "v3", credentials=creds)
        service.files().get(fileId=folder_id, fields="id, name").execute()
    except Exception as e:
        sa_email = getattr(creds, "service_account_email", None)
        if sa_email:
            return (
                f"Cannot access Drive folder {folder_id}. "
                f"Share the folder with {sa_email} (Viewer access) in Google Drive, "
                "then try again."
            )
        return f"Cannot access Drive folder {folder_id}: {e}"

    return None


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
