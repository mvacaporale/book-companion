"""Tests for Google Drive credential chain and access verification."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from google.auth.exceptions import DefaultCredentialsError

from book_companion.google_drive.auth import (
    get_credentials,
    verify_drive_access,
)


def _service_account_creds(email: str = "books-sa@project.iam.gserviceaccount.com"):
    creds = MagicMock()
    creds.service_account_email = email
    return creds


class TestGetCredentials:
    """Credential chain: service account (ADC) first, local token file second."""

    def test_returns_service_account_credentials_from_adc(self, tmp_path: Path):
        sa_creds = _service_account_creds()
        with patch("google.auth.default", return_value=(sa_creds, "project")):
            creds = get_credentials(token_path=tmp_path / "missing_token.json")
        assert creds is sa_creds

    def test_no_adc_and_no_token_file_returns_none(self, tmp_path: Path):
        with patch("google.auth.default", side_effect=DefaultCredentialsError):
            creds = get_credentials(token_path=tmp_path / "missing_token.json")
        assert creds is None

    def test_user_adc_without_drive_scopes_is_ignored(self, tmp_path: Path):
        # End-user ADC creds (no service_account_email) can't be Drive-scoped
        user_creds = MagicMock(spec=[])
        with patch("google.auth.default", return_value=(user_creds, "project")):
            creds = get_credentials(token_path=tmp_path / "missing_token.json")
        assert creds is None


class TestVerifyDriveAccess:
    def test_no_credentials_returns_setup_help(self):
        with patch(
            "book_companion.google_drive.auth.get_credentials", return_value=None
        ):
            error = verify_drive_access(folder_id="folder123")
        assert error is not None
        assert "not configured" in error

    def test_no_folder_configured_returns_help(self):
        with patch(
            "book_companion.google_drive.auth.get_credentials",
            return_value=_service_account_creds(),
        ), patch(
            "book_companion.google_drive.auth.get_config", return_value={}
        ):
            error = verify_drive_access()
        assert error is not None
        assert "GOOGLE_DRIVE_FOLDER_ID" in error

    def test_unshared_folder_error_names_service_account_email(self):
        sa_creds = _service_account_creds("books-sa@project.iam.gserviceaccount.com")
        mock_build = MagicMock()
        mock_build.return_value.files.return_value.get.return_value.execute.side_effect = (
            Exception("404 not found")
        )
        with patch(
            "book_companion.google_drive.auth.get_credentials", return_value=sa_creds
        ), patch("googleapiclient.discovery.build", mock_build):
            error = verify_drive_access(folder_id="folder123")
        assert error is not None
        assert "books-sa@project.iam.gserviceaccount.com" in error
        assert "folder123" in error

    def test_accessible_folder_returns_none(self):
        mock_build = MagicMock()
        mock_build.return_value.files.return_value.get.return_value.execute.return_value = {
            "id": "folder123",
            "name": "Books",
        }
        with patch(
            "book_companion.google_drive.auth.get_credentials",
            return_value=_service_account_creds(),
        ), patch("googleapiclient.discovery.build", mock_build):
            error = verify_drive_access(folder_id="folder123")
        assert error is None
