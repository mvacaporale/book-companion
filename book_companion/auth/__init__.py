"""OAuth 2.1 authentication module for MCP server."""

from book_companion.auth.config import OAuthConfig
from book_companion.auth.middleware import oauth_middleware, require_auth
from book_companion.auth.server import create_oauth_routes

__all__ = [
    "OAuthConfig",
    "oauth_middleware",
    "require_auth",
    "create_oauth_routes",
]
