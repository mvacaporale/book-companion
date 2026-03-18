"""Tests for OAuth 2.1 endpoints."""

import pytest
from unittest.mock import patch, MagicMock

from book_companion.auth.config import OAuthConfig
from book_companion.auth.models import OAuthClient, AuthorizationCode, AccessToken, generate_token
from book_companion.auth.store import OAuthStore


class TestOAuthModels:
    """Test cases for OAuth data models."""

    def test_generate_token(self) -> None:
        """Generated tokens should be unique and URL-safe."""
        token1 = generate_token(32)
        token2 = generate_token(32)
        assert token1 != token2
        assert len(token1) > 20  # Base64 encoding adds some length

    def test_oauth_client_to_dict(self) -> None:
        """OAuthClient should serialize to dict correctly."""
        client = OAuthClient(
            client_id="test-id",
            client_secret="test-secret",
            client_name="Test Client",
            redirect_uris=["http://localhost/callback"],
        )
        data = client.to_dict()
        assert data["client_id"] == "test-id"
        assert data["client_secret"] == "test-secret"
        assert data["redirect_uris"] == ["http://localhost/callback"]

    def test_oauth_client_from_dict(self) -> None:
        """OAuthClient should deserialize from dict correctly."""
        data = {
            "client_id": "test-id",
            "client_secret": "test-secret",
            "client_name": "Test Client",
            "redirect_uris": ["http://localhost/callback"],
        }
        client = OAuthClient.from_dict(data)
        assert client.client_id == "test-id"
        assert client.client_secret == "test-secret"

    def test_authorization_code_expiry(self) -> None:
        """Authorization code expiry should work correctly."""
        import time

        # Not expired
        code = AuthorizationCode(
            code="test-code",
            client_id="client-id",
            redirect_uri="http://localhost/callback",
            scope="mcp",
            expires_at=time.time() + 600,
        )
        assert not code.is_expired()

        # Expired
        code_expired = AuthorizationCode(
            code="test-code",
            client_id="client-id",
            redirect_uri="http://localhost/callback",
            scope="mcp",
            expires_at=time.time() - 100,
        )
        assert code_expired.is_expired()

    def test_access_token_expiry(self) -> None:
        """Access token expiry should work correctly."""
        import time

        token = AccessToken(
            token="test-token",
            client_id="client-id",
            scope="mcp",
            expires_at=time.time() + 3600,
            refresh_token="refresh-token",
            refresh_expires_at=time.time() + 604800,
        )
        assert not token.is_expired()
        assert not token.is_refresh_expired()

        # Expired access, valid refresh
        token_access_expired = AccessToken(
            token="test-token",
            client_id="client-id",
            scope="mcp",
            expires_at=time.time() - 100,
            refresh_token="refresh-token",
            refresh_expires_at=time.time() + 604800,
        )
        assert token_access_expired.is_expired()
        assert not token_access_expired.is_refresh_expired()


class TestOAuthStore:
    """Test cases for OAuth storage."""

    @pytest.fixture
    def store(self, mock_bookrc_dir) -> OAuthStore:
        """Create a store with a temp directory."""
        with patch("book_companion.auth.store.get_oauth_config") as mock_config:
            config = OAuthConfig()
            config.data_dir = mock_bookrc_dir
            mock_config.return_value = config
            return OAuthStore()

    def test_save_and_get_client(self, store: OAuthStore) -> None:
        """Saving and retrieving a client should work."""
        client = OAuthClient(
            client_id="test-client",
            client_secret="secret",
            client_name="Test",
            redirect_uris=["http://localhost/callback"],
        )
        store.save_client(client)

        retrieved = store.get_client("test-client")
        assert retrieved is not None
        assert retrieved.client_id == "test-client"
        assert retrieved.client_secret == "secret"

    def test_get_nonexistent_client(self, store: OAuthStore) -> None:
        """Getting a nonexistent client should return None."""
        assert store.get_client("nonexistent") is None

    def test_delete_client(self, store: OAuthStore) -> None:
        """Deleting a client should work."""
        client = OAuthClient(
            client_id="to-delete",
            client_secret="secret",
            client_name="Test",
        )
        store.save_client(client)
        assert store.get_client("to-delete") is not None

        store.delete_client("to-delete")
        assert store.get_client("to-delete") is None

    def test_save_and_get_auth_code(self, store: OAuthStore) -> None:
        """Saving and retrieving an auth code should work."""
        import time

        code = AuthorizationCode(
            code="test-code",
            client_id="client-id",
            redirect_uri="http://localhost/callback",
            scope="mcp",
            expires_at=time.time() + 600,
        )
        store.save_auth_code(code)

        retrieved = store.get_auth_code("test-code")
        assert retrieved is not None
        assert retrieved.code == "test-code"
        assert retrieved.client_id == "client-id"

    def test_delete_auth_code(self, store: OAuthStore) -> None:
        """Auth codes should be deleted after use (single-use)."""
        import time

        code = AuthorizationCode(
            code="single-use-code",
            client_id="client-id",
            redirect_uri="http://localhost/callback",
            scope="mcp",
            expires_at=time.time() + 600,
        )
        store.save_auth_code(code)
        assert store.get_auth_code("single-use-code") is not None

        store.delete_auth_code("single-use-code")
        assert store.get_auth_code("single-use-code") is None

    def test_save_and_get_token(self, store: OAuthStore) -> None:
        """Saving and retrieving a token should work."""
        import time

        token = AccessToken(
            token="access-token",
            client_id="client-id",
            scope="mcp",
            expires_at=time.time() + 3600,
            refresh_token="refresh-token",
            refresh_expires_at=time.time() + 604800,
        )
        store.save_token(token)

        retrieved = store.get_token("access-token")
        assert retrieved is not None
        assert retrieved.token == "access-token"
        assert retrieved.refresh_token == "refresh-token"

    def test_get_token_by_refresh(self, store: OAuthStore) -> None:
        """Getting a token by refresh token should work."""
        import time

        token = AccessToken(
            token="access-token",
            client_id="client-id",
            scope="mcp",
            expires_at=time.time() + 3600,
            refresh_token="unique-refresh-token",
            refresh_expires_at=time.time() + 604800,
        )
        store.save_token(token)

        retrieved = store.get_token_by_refresh("unique-refresh-token")
        assert retrieved is not None
        assert retrieved.token == "access-token"

    def test_cleanup_expired_codes(self, store: OAuthStore) -> None:
        """Expired auth codes should be cleaned up."""
        import time

        # Add expired code
        expired_code = AuthorizationCode(
            code="expired-code",
            client_id="client-id",
            redirect_uri="http://localhost/callback",
            scope="mcp",
            expires_at=time.time() - 100,
        )
        store.save_auth_code(expired_code)

        # Add valid code
        valid_code = AuthorizationCode(
            code="valid-code",
            client_id="client-id",
            redirect_uri="http://localhost/callback",
            scope="mcp",
            expires_at=time.time() + 600,
        )
        store.save_auth_code(valid_code)

        # Cleanup
        cleaned = store.cleanup_expired_codes()
        assert cleaned == 1
        assert store.get_auth_code("expired-code") is None
        assert store.get_auth_code("valid-code") is not None
