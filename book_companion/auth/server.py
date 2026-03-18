"""OAuth 2.1 server endpoints for MCP authentication.

Implements the MCP third-party authorization pattern:
- MCP server acts as OAuth authorization server to Claude
- Delegates actual authentication to Google OAuth (already configured for Drive)
- Issues bound access tokens that Claude uses for subsequent requests

Required Endpoints (per MCP Authorization Spec):
- /.well-known/oauth-authorization-server: Metadata discovery
- /authorize: Redirect user to Google OAuth
- /token: Exchange auth code for access token
- /register: Dynamic client registration
"""

import base64
import hashlib
import json
import logging
import time
import urllib.parse
from typing import Optional

logger = logging.getLogger(__name__)

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Route

from book_companion.auth.config import get_oauth_config
from book_companion.auth.models import (
    OAuthClient,
    AuthorizationCode,
    AccessToken,
    generate_token,
)
from book_companion.auth.store import get_oauth_store
from book_companion.google_drive.auth import get_credentials


def get_issuer(request: Request) -> str:
    """Get the OAuth issuer URL from request or config."""
    config = get_oauth_config()
    if config.issuer:
        return config.issuer
    # Auto-detect from request
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("host", request.url.netloc)
    return f"{scheme}://{host}"


async def oauth_metadata(request: Request) -> JSONResponse:
    """OAuth 2.0 Authorization Server Metadata (RFC 8414).

    Endpoint: /.well-known/oauth-authorization-server
    """
    issuer = get_issuer(request)
    config = get_oauth_config()

    metadata = {
        "issuer": issuer,
        "authorization_endpoint": f"{issuer}/authorize",
        "token_endpoint": f"{issuer}/token",
        "registration_endpoint": f"{issuer}/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["client_secret_post", "client_secret_basic"],
        "scopes_supported": ["mcp"],
        "service_documentation": f"{issuer}/docs",
    }

    return JSONResponse(metadata)


async def protected_resource_metadata(request: Request) -> JSONResponse:
    """OAuth 2.0 Protected Resource Metadata (RFC 9728 / MCP spec).

    Endpoint: /.well-known/oauth-protected-resource

    This tells clients where to find the authorization server for this resource.
    """
    issuer = get_issuer(request)

    metadata = {
        "resource": f"{issuer}/mcp",
        "authorization_servers": [issuer],
        "scopes_supported": ["mcp"],
        "bearer_methods_supported": ["header"],
    }

    return JSONResponse(metadata)


async def register_client(request: Request) -> JSONResponse:
    """Dynamic Client Registration (RFC 7591).

    Endpoint: POST /register
    """
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(
            {"error": "invalid_request", "error_description": "Invalid JSON body"},
            status_code=400,
        )

    # Validate required fields
    client_name = body.get("client_name")
    if not client_name:
        return JSONResponse(
            {"error": "invalid_request", "error_description": "client_name is required"},
            status_code=400,
        )

    redirect_uris = body.get("redirect_uris", [])
    if not redirect_uris:
        return JSONResponse(
            {"error": "invalid_request", "error_description": "redirect_uris is required"},
            status_code=400,
        )

    # Generate client credentials
    client = OAuthClient(
        client_id=generate_token(16),
        client_secret=generate_token(32),
        client_name=client_name,
        redirect_uris=redirect_uris,
        grant_types=body.get("grant_types", ["authorization_code", "refresh_token"]),
        response_types=body.get("response_types", ["code"]),
        scope=body.get("scope", "mcp"),
    )

    # Save client
    store = get_oauth_store()
    store.save_client(client)

    # Return registration response
    return JSONResponse(
        {
            "client_id": client.client_id,
            "client_secret": client.client_secret,
            "client_name": client.client_name,
            "redirect_uris": client.redirect_uris,
            "grant_types": client.grant_types,
            "response_types": client.response_types,
            "scope": client.scope,
            "client_id_issued_at": int(client.created_at),
        },
        status_code=201,
    )


async def authorize(request: Request) -> Response:
    """Authorization endpoint.

    Endpoint: GET /authorize

    This endpoint validates the OAuth parameters and checks if Google Drive
    credentials are available. If credentials exist, it issues an authorization
    code immediately. Otherwise, it returns an error.

    Query Parameters:
        response_type: Must be "code"
        client_id: Registered client ID
        redirect_uri: Must match registered redirect URI
        scope: Requested scope (default: mcp)
        state: CSRF protection (recommended)
        code_challenge: PKCE challenge (recommended)
        code_challenge_method: Must be "S256" if code_challenge provided
    """
    # Extract parameters
    response_type = request.query_params.get("response_type")
    client_id = request.query_params.get("client_id")
    redirect_uri = request.query_params.get("redirect_uri")
    scope = request.query_params.get("scope", "mcp")
    state = request.query_params.get("state")
    code_challenge = request.query_params.get("code_challenge")
    code_challenge_method = request.query_params.get("code_challenge_method")

    logger.info(f"Authorize endpoint called. client_id: {client_id}, redirect_uri: {redirect_uri}")
    logger.info(f"response_type: {response_type}, scope: {scope}, state: {state[:20] if state else None}...")

    # Validate response_type
    if response_type != "code":
        return _error_redirect(
            redirect_uri, "unsupported_response_type",
            "Only 'code' response type is supported", state
        )

    # Validate client
    store = get_oauth_store()
    client = store.get_client(client_id) if client_id else None
    if not client:
        return _error_redirect(
            redirect_uri, "invalid_client",
            "Unknown client_id", state
        )

    # Validate redirect_uri
    if redirect_uri not in client.redirect_uris:
        return _error_redirect(
            redirect_uri, "invalid_request",
            "redirect_uri not registered for this client", state
        )

    # Validate PKCE if provided
    if code_challenge and code_challenge_method != "S256":
        return _error_redirect(
            redirect_uri, "invalid_request",
            "Only S256 code_challenge_method is supported", state
        )

    # Check if Google Drive credentials are available
    # (This is our "authentication" - if the user has Drive access, they're authorized)
    creds = get_credentials()
    if not creds or not creds.valid:
        return _error_redirect(
            redirect_uri, "access_denied",
            "Google Drive authentication required. Run 'bookrc setup-drive' first.", state
        )

    # Generate authorization code
    config = get_oauth_config()
    auth_code = AuthorizationCode(
        code=generate_token(32),
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=scope,
        expires_at=time.time() + 600,  # 10 minute expiry
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
    )
    store.save_auth_code(auth_code)

    # Redirect with authorization code
    params = {"code": auth_code.code}
    if state:
        params["state"] = state
    redirect_url = f"{redirect_uri}?{urllib.parse.urlencode(params)}"
    logger.info(f"Redirecting with auth code to: {redirect_uri} (code: {auth_code.code[:8]}...)")
    return RedirectResponse(redirect_url, status_code=302)


def _error_redirect(
    redirect_uri: Optional[str],
    error: str,
    description: str,
    state: Optional[str],
) -> Response:
    """Create an error redirect response."""
    if not redirect_uri:
        return JSONResponse(
            {"error": error, "error_description": description},
            status_code=400,
        )
    params = {"error": error, "error_description": description}
    if state:
        params["state"] = state
    redirect_url = f"{redirect_uri}?{urllib.parse.urlencode(params)}"
    return RedirectResponse(redirect_url, status_code=302)


async def token(request: Request) -> JSONResponse:
    """Token endpoint.

    Endpoint: POST /token

    Supports:
    - authorization_code: Exchange auth code for access token
    - refresh_token: Refresh an expired access token

    Form Parameters (or JSON body):
        grant_type: "authorization_code" or "refresh_token"
        code: Authorization code (for authorization_code grant)
        redirect_uri: Must match original request (for authorization_code grant)
        client_id: Client ID
        client_secret: Client secret
        code_verifier: PKCE verifier (if code_challenge was used)
        refresh_token: Refresh token (for refresh_token grant)
    """
    # Log incoming request for debugging
    content_type = request.headers.get("content-type", "")
    logger.info(f"Token endpoint called. Content-Type: {content_type}")
    logger.info(f"Headers: {dict(request.headers)}")

    # Parse form data or JSON body (Claude may send either)
    if "application/json" in content_type:
        try:
            body = await request.body()
            logger.info(f"Raw JSON body: {body.decode('utf-8', errors='replace')}")
            form = json.loads(body)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return JSONResponse(
                {"error": "invalid_request", "error_description": "Invalid JSON body"},
                status_code=400,
            )
    else:
        form = await request.form()
        logger.info(f"Form data keys: {list(form.keys())}")

    grant_type = form.get("grant_type")
    logger.info(f"grant_type: {grant_type}")
    client_id = form.get("client_id")
    client_secret = form.get("client_secret")

    # Also check Authorization header for Basic auth (client_secret_basic)
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Basic ") and (not client_id or not client_secret):
        import base64
        try:
            decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
            if ":" in decoded:
                header_client_id, header_client_secret = decoded.split(":", 1)
                client_id = client_id or header_client_id
                client_secret = client_secret or header_client_secret
        except Exception:
            pass  # Fall through to invalid_client error

    # Validate client
    store = get_oauth_store()
    client = store.get_client(client_id) if client_id else None
    if not client or client.client_secret != client_secret:
        return JSONResponse(
            {"error": "invalid_client", "error_description": "Invalid client credentials"},
            status_code=401,
        )

    config = get_oauth_config()

    if grant_type == "authorization_code":
        return await _handle_authorization_code(request, form, client, store, config)
    elif grant_type == "refresh_token":
        return await _handle_refresh_token(request, form, client, store, config)
    else:
        return JSONResponse(
            {"error": "unsupported_grant_type", "error_description": "Unsupported grant type"},
            status_code=400,
        )


async def _handle_authorization_code(
    request: Request,
    form: dict,
    client: OAuthClient,
    store,
    config,
) -> JSONResponse:
    """Handle authorization_code grant type."""
    code = form.get("code")
    redirect_uri = form.get("redirect_uri")
    code_verifier = form.get("code_verifier")

    # Get and validate auth code
    auth_code = store.get_auth_code(code) if code else None
    if not auth_code:
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "Invalid authorization code"},
            status_code=400,
        )

    # Check expiration
    if auth_code.is_expired():
        store.delete_auth_code(code)
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "Authorization code expired"},
            status_code=400,
        )

    # Validate client_id matches
    if auth_code.client_id != client.client_id:
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "Client ID mismatch"},
            status_code=400,
        )

    # Validate redirect_uri matches
    if auth_code.redirect_uri != redirect_uri:
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "Redirect URI mismatch"},
            status_code=400,
        )

    # Validate PKCE if code_challenge was used
    if auth_code.code_challenge:
        if not code_verifier:
            return JSONResponse(
                {"error": "invalid_grant", "error_description": "code_verifier required"},
                status_code=400,
            )
        # Verify S256 challenge
        verifier_hash = hashlib.sha256(code_verifier.encode()).digest()
        expected = base64.urlsafe_b64encode(verifier_hash).rstrip(b"=").decode()
        if expected != auth_code.code_challenge:
            return JSONResponse(
                {"error": "invalid_grant", "error_description": "Invalid code_verifier"},
                status_code=400,
            )

    # Delete auth code (single use)
    store.delete_auth_code(code)

    # Generate access token
    access_token = AccessToken(
        token=generate_token(32),
        client_id=client.client_id,
        scope=auth_code.scope,
        expires_at=time.time() + config.access_token_expiry,
        refresh_token=generate_token(32),
        refresh_expires_at=time.time() + config.refresh_token_expiry,
    )
    store.save_token(access_token)

    return JSONResponse({
        "access_token": access_token.token,
        "token_type": "Bearer",
        "expires_in": config.access_token_expiry,
        "refresh_token": access_token.refresh_token,
        "scope": access_token.scope,
    })


async def _handle_refresh_token(
    request: Request,
    form: dict,
    client: OAuthClient,
    store,
    config,
) -> JSONResponse:
    """Handle refresh_token grant type."""
    refresh_token = form.get("refresh_token")

    # Find token by refresh token
    old_token = store.get_token_by_refresh(refresh_token) if refresh_token else None
    if not old_token:
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "Invalid refresh token"},
            status_code=400,
        )

    # Check refresh token expiration
    if old_token.is_refresh_expired():
        store.delete_token(old_token.token)
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "Refresh token expired"},
            status_code=400,
        )

    # Validate client_id matches
    if old_token.client_id != client.client_id:
        return JSONResponse(
            {"error": "invalid_grant", "error_description": "Client ID mismatch"},
            status_code=400,
        )

    # Delete old token
    store.delete_token(old_token.token)

    # Generate new access token
    new_token = AccessToken(
        token=generate_token(32),
        client_id=client.client_id,
        scope=old_token.scope,
        expires_at=time.time() + config.access_token_expiry,
        refresh_token=generate_token(32),
        refresh_expires_at=time.time() + config.refresh_token_expiry,
    )
    store.save_token(new_token)

    return JSONResponse({
        "access_token": new_token.token,
        "token_type": "Bearer",
        "expires_in": config.access_token_expiry,
        "refresh_token": new_token.refresh_token,
        "scope": new_token.scope,
    })


def create_oauth_routes() -> list[Route]:
    """Create OAuth endpoint routes for the MCP server."""
    return [
        Route("/.well-known/oauth-authorization-server", oauth_metadata, methods=["GET"]),
        Route("/.well-known/oauth-protected-resource", protected_resource_metadata, methods=["GET"]),
        Route("/register", register_client, methods=["POST"]),
        Route("/authorize", authorize, methods=["GET"]),
        Route("/token", token, methods=["POST"]),
    ]
