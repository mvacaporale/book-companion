"""OAuth data models."""

import secrets
import time
from dataclasses import dataclass, field
from typing import Optional


def generate_token(length: int = 32) -> str:
    """Generate a cryptographically secure random token."""
    return secrets.token_urlsafe(length)


@dataclass
class OAuthClient:
    """OAuth 2.1 client registration.

    Per MCP spec, supports dynamic client registration.
    """

    client_id: str
    client_secret: str
    client_name: str
    redirect_uris: list[str] = field(default_factory=list)
    grant_types: list[str] = field(default_factory=lambda: ["authorization_code", "refresh_token"])
    response_types: list[str] = field(default_factory=lambda: ["code"])
    scope: str = "mcp"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "client_name": self.client_name,
            "redirect_uris": self.redirect_uris,
            "grant_types": self.grant_types,
            "response_types": self.response_types,
            "scope": self.scope,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OAuthClient":
        """Create from dictionary."""
        return cls(
            client_id=data["client_id"],
            client_secret=data["client_secret"],
            client_name=data["client_name"],
            redirect_uris=data.get("redirect_uris", []),
            grant_types=data.get("grant_types", ["authorization_code", "refresh_token"]),
            response_types=data.get("response_types", ["code"]),
            scope=data.get("scope", "mcp"),
            created_at=data.get("created_at", time.time()),
        )


@dataclass
class AuthorizationCode:
    """OAuth authorization code (short-lived)."""

    code: str
    client_id: str
    redirect_uri: str
    scope: str
    expires_at: float
    code_challenge: Optional[str] = None
    code_challenge_method: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if the authorization code has expired."""
        return time.time() > self.expires_at

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "code": self.code,
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": self.scope,
            "expires_at": self.expires_at,
            "code_challenge": self.code_challenge,
            "code_challenge_method": self.code_challenge_method,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AuthorizationCode":
        """Create from dictionary."""
        return cls(
            code=data["code"],
            client_id=data["client_id"],
            redirect_uri=data["redirect_uri"],
            scope=data["scope"],
            expires_at=data["expires_at"],
            code_challenge=data.get("code_challenge"),
            code_challenge_method=data.get("code_challenge_method"),
        )


@dataclass
class AccessToken:
    """OAuth access token."""

    token: str
    client_id: str
    scope: str
    expires_at: float
    refresh_token: Optional[str] = None
    refresh_expires_at: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if the access token has expired."""
        return time.time() > self.expires_at

    def is_refresh_expired(self) -> bool:
        """Check if the refresh token has expired."""
        if self.refresh_expires_at is None:
            return True
        return time.time() > self.refresh_expires_at

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "token": self.token,
            "client_id": self.client_id,
            "scope": self.scope,
            "expires_at": self.expires_at,
            "refresh_token": self.refresh_token,
            "refresh_expires_at": self.refresh_expires_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AccessToken":
        """Create from dictionary."""
        return cls(
            token=data["token"],
            client_id=data["client_id"],
            scope=data["scope"],
            expires_at=data["expires_at"],
            refresh_token=data.get("refresh_token"),
            refresh_expires_at=data.get("refresh_expires_at"),
        )
