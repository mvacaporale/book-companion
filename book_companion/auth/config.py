"""OAuth configuration and constants."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class OAuthConfig:
    """OAuth 2.1 configuration for MCP server.

    Environment Variables:
        MCP_OAUTH_ENABLED: Enable OAuth authentication (default: false)
        MCP_OAUTH_ISSUER: OAuth issuer URL (default: auto-detected from request)
        MCP_OAUTH_TOKEN_EXPIRY: Access token expiry in seconds (default: 3600)
        MCP_OAUTH_REFRESH_TOKEN_EXPIRY: Refresh token expiry in seconds (default: 604800)
    """

    # Whether OAuth is enabled
    enabled: bool = field(default_factory=lambda: os.environ.get(
        "MCP_OAUTH_ENABLED", ""
    ).lower() in ("true", "1", "yes"))

    # OAuth issuer URL (auto-detected from Cloud Run URL if not set)
    issuer: Optional[str] = field(default_factory=lambda: os.environ.get("MCP_OAUTH_ISSUER"))

    # Token expiration times (in seconds)
    access_token_expiry: int = field(default_factory=lambda: int(
        os.environ.get("MCP_OAUTH_TOKEN_EXPIRY", "3600")
    ))
    refresh_token_expiry: int = field(default_factory=lambda: int(
        os.environ.get("MCP_OAUTH_REFRESH_TOKEN_EXPIRY", "604800")
    ))

    # Storage paths
    data_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("BOOKRC_DB_PATH", str(Path.home() / ".bookrc"))
    ))

    @property
    def clients_path(self) -> Path:
        """Path to OAuth clients storage file."""
        return self.data_dir / "oauth_clients.json"

    @property
    def tokens_path(self) -> Path:
        """Path to OAuth tokens storage file."""
        return self.data_dir / "oauth_tokens.json"

    @property
    def auth_codes_path(self) -> Path:
        """Path to authorization codes storage file."""
        return self.data_dir / "oauth_auth_codes.json"


# Global config instance
_config: Optional[OAuthConfig] = None


def get_oauth_config() -> OAuthConfig:
    """Get the global OAuth configuration."""
    global _config
    if _config is None:
        _config = OAuthConfig()
    return _config
