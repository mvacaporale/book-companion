"""Storage for OAuth clients, tokens, and authorization codes."""

import json
import os
from pathlib import Path
from typing import Optional

from book_companion.auth.config import get_oauth_config
from book_companion.auth.models import OAuthClient, AuthorizationCode, AccessToken


class OAuthStore:
    """File-based storage for OAuth data.

    Stores clients, authorization codes, and access tokens in JSON files.
    For production, consider using a database instead.
    """

    def __init__(self) -> None:
        self.config = get_oauth_config()
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Ensure storage directories exist."""
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

    def _load_json(self, path: Path) -> dict:
        """Load JSON file, returning empty dict if not exists."""
        if not path.exists():
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def _save_json(self, path: Path, data: dict) -> None:
        """Save data to JSON file with secure permissions."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        # Set restrictive permissions
        os.chmod(path, 0o600)

    # =========================================================================
    # Client Operations
    # =========================================================================

    def save_client(self, client: OAuthClient) -> None:
        """Save an OAuth client."""
        clients = self._load_json(self.config.clients_path)
        clients[client.client_id] = client.to_dict()
        self._save_json(self.config.clients_path, clients)

    def get_client(self, client_id: str) -> Optional[OAuthClient]:
        """Get an OAuth client by ID."""
        clients = self._load_json(self.config.clients_path)
        data = clients.get(client_id)
        if data:
            return OAuthClient.from_dict(data)
        return None

    def delete_client(self, client_id: str) -> bool:
        """Delete an OAuth client."""
        clients = self._load_json(self.config.clients_path)
        if client_id in clients:
            del clients[client_id]
            self._save_json(self.config.clients_path, clients)
            return True
        return False

    # =========================================================================
    # Authorization Code Operations
    # =========================================================================

    def save_auth_code(self, code: AuthorizationCode) -> None:
        """Save an authorization code."""
        codes = self._load_json(self.config.auth_codes_path)
        codes[code.code] = code.to_dict()
        self._save_json(self.config.auth_codes_path, codes)

    def get_auth_code(self, code: str) -> Optional[AuthorizationCode]:
        """Get an authorization code."""
        codes = self._load_json(self.config.auth_codes_path)
        data = codes.get(code)
        if data:
            return AuthorizationCode.from_dict(data)
        return None

    def delete_auth_code(self, code: str) -> bool:
        """Delete an authorization code (single use)."""
        codes = self._load_json(self.config.auth_codes_path)
        if code in codes:
            del codes[code]
            self._save_json(self.config.auth_codes_path, codes)
            return True
        return False

    def cleanup_expired_codes(self) -> int:
        """Remove expired authorization codes."""
        codes = self._load_json(self.config.auth_codes_path)
        import time

        expired = [
            code_str
            for code_str, data in codes.items()
            if data.get("expires_at", 0) < time.time()
        ]
        for code_str in expired:
            del codes[code_str]
        if expired:
            self._save_json(self.config.auth_codes_path, codes)
        return len(expired)

    # =========================================================================
    # Access Token Operations
    # =========================================================================

    def save_token(self, token: AccessToken) -> None:
        """Save an access token."""
        tokens = self._load_json(self.config.tokens_path)
        tokens[token.token] = token.to_dict()
        self._save_json(self.config.tokens_path, tokens)

    def get_token(self, token: str) -> Optional[AccessToken]:
        """Get an access token."""
        tokens = self._load_json(self.config.tokens_path)
        data = tokens.get(token)
        if data:
            return AccessToken.from_dict(data)
        return None

    def get_token_by_refresh(self, refresh_token: str) -> Optional[AccessToken]:
        """Get an access token by its refresh token."""
        tokens = self._load_json(self.config.tokens_path)
        for data in tokens.values():
            if data.get("refresh_token") == refresh_token:
                return AccessToken.from_dict(data)
        return None

    def delete_token(self, token: str) -> bool:
        """Delete an access token."""
        tokens = self._load_json(self.config.tokens_path)
        if token in tokens:
            del tokens[token]
            self._save_json(self.config.tokens_path, tokens)
            return True
        return False

    def cleanup_expired_tokens(self) -> int:
        """Remove expired access tokens."""
        tokens = self._load_json(self.config.tokens_path)
        import time

        # Remove tokens where both access and refresh are expired
        expired = []
        for token_str, data in tokens.items():
            access_expired = data.get("expires_at", 0) < time.time()
            refresh_expires = data.get("refresh_expires_at")
            refresh_expired = refresh_expires is None or refresh_expires < time.time()
            if access_expired and refresh_expired:
                expired.append(token_str)

        for token_str in expired:
            del tokens[token_str]
        if expired:
            self._save_json(self.config.tokens_path, tokens)
        return len(expired)


# Global store instance
_store: Optional[OAuthStore] = None


def get_oauth_store() -> OAuthStore:
    """Get the global OAuth store."""
    global _store
    if _store is None:
        _store = OAuthStore()
    return _store
