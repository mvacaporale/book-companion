"""OAuth middleware for request authentication."""

from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from book_companion.auth.config import get_oauth_config
from book_companion.auth.store import get_oauth_store


# Paths that don't require authentication
PUBLIC_PATHS = {
    "/.well-known/oauth-authorization-server",
    "/.well-known/oauth-protected-resource",
    "/register",
    "/authorize",
    "/token",
    "/openapi.json",
    "/docs",
    "/health",
}


def is_public_path(path: str) -> bool:
    """Check if a path is public (doesn't require auth)."""
    return path in PUBLIC_PATHS


def validate_bearer_token(auth_header: Optional[str]) -> Optional[str]:
    """Extract and validate a Bearer token from Authorization header.

    Args:
        auth_header: The Authorization header value

    Returns:
        The client_id if token is valid, None otherwise
    """
    if not auth_header:
        return None

    parts = auth_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None

    token = parts[1]
    store = get_oauth_store()
    access_token = store.get_token(token)

    if not access_token:
        return None

    if access_token.is_expired():
        return None

    return access_token.client_id


class OAuthMiddleware(BaseHTTPMiddleware):
    """Middleware that validates OAuth tokens on protected endpoints.

    When OAuth is enabled (MCP_OAUTH_ENABLED=true), all requests to
    non-public paths must include a valid Bearer token.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        config = get_oauth_config()

        # If OAuth is disabled, allow all requests
        if not config.enabled:
            return await call_next(request)

        # Allow public paths without authentication
        if is_public_path(request.url.path):
            return await call_next(request)

        # Validate Bearer token
        auth_header = request.headers.get("authorization")
        client_id = validate_bearer_token(auth_header)

        if not client_id:
            return JSONResponse(
                {
                    "error": "invalid_token",
                    "error_description": "Missing or invalid access token",
                },
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="mcp"'},
            )

        # Add client_id to request state for use in handlers
        request.state.oauth_client_id = client_id

        return await call_next(request)


def oauth_middleware(app):
    """Create OAuth middleware for a Starlette app."""
    return OAuthMiddleware(app)


def require_auth(request: Request) -> Optional[str]:
    """Check if request is authenticated and return client_id.

    Use this in route handlers that need to verify authentication
    even when OAuth middleware is disabled.

    Args:
        request: The Starlette request object

    Returns:
        The client_id if authenticated, None otherwise

    Example:
        @app.route("/protected")
        async def protected(request):
            client_id = require_auth(request)
            if not client_id:
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            return JSONResponse({"client": client_id})
    """
    config = get_oauth_config()

    # If OAuth is disabled, return a placeholder
    if not config.enabled:
        return "local"

    # Check request state (set by middleware)
    if hasattr(request.state, "oauth_client_id"):
        return request.state.oauth_client_id

    # Validate token directly (for cases where middleware wasn't applied)
    auth_header = request.headers.get("authorization")
    return validate_bearer_token(auth_header)
