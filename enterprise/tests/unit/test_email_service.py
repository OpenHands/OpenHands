"""Tests for email service."""

import os
from unittest.mock import MagicMock, patch

from server.services.email_service import (
    DEFAULT_WEB_HOST,
    EmailService,
)


class TestEmailServiceInvitationUrl:
    """Test cases for invitation URL generation."""

    def test_invitation_url_uses_correct_endpoint(self):
        """Test that invitation URL points to the correct API endpoint."""
        mock_response = MagicMock()
        mock_response.get.return_value = 'test-email-id'

        with (
            patch.dict(os.environ, {'RESEND_API_KEY': 'test-key'}),
            patch('server.services.email_service.RESEND_AVAILABLE', True),
            patch('server.services.email_service.resend') as mock_resend,
        ):
            mock_resend.Emails.send.return_value = mock_response

            EmailService.send_invitation_email(
                to_email='test@example.com',
                org_name='Test Org',
                inviter_name='Inviter',
                role_name='member',
                invitation_token='inv-test-token-12345',
                invitation_id=1,
            )

            # Get the call arguments
            call_args = mock_resend.Emails.send.call_args
            email_params = call_args[0][0]

            # Verify the URL in the email HTML contains the correct endpoint
            assert (
                '/api/organizations/members/invite/accept?token='
                in email_params['html']
            )
            assert 'inv-test-token-12345' in email_params['html']

    def test_invitation_url_uses_web_host_env_var(self):
        """Test that invitation URL uses WEB_HOST environment variable."""
        custom_host = 'https://custom.example.com'
        mock_response = MagicMock()
        mock_response.get.return_value = 'test-email-id'

        with (
            patch.dict(
                os.environ,
                {'RESEND_API_KEY': 'test-key', 'WEB_HOST': custom_host},
            ),
            patch('server.services.email_service.RESEND_AVAILABLE', True),
            patch('server.services.email_service.resend') as mock_resend,
        ):
            mock_resend.Emails.send.return_value = mock_response

            EmailService.send_invitation_email(
                to_email='test@example.com',
                org_name='Test Org',
                inviter_name='Inviter',
                role_name='member',
                invitation_token='inv-test-token-12345',
                invitation_id=1,
            )

            call_args = mock_resend.Emails.send.call_args
            email_params = call_args[0][0]

            expected_url = f'{custom_host}/api/organizations/members/invite/accept?token=inv-test-token-12345'
            assert expected_url in email_params['html']

    def test_invitation_url_uses_default_host_when_env_not_set(self):
        """Test that invitation URL falls back to DEFAULT_WEB_HOST when env not set."""
        mock_response = MagicMock()
        mock_response.get.return_value = 'test-email-id'

        env_without_web_host = {'RESEND_API_KEY': 'test-key'}
        # Remove WEB_HOST if it exists
        env_without_web_host.pop('WEB_HOST', None)

        with (
            patch.dict(os.environ, env_without_web_host, clear=True),
            patch('server.services.email_service.RESEND_AVAILABLE', True),
            patch('server.services.email_service.resend') as mock_resend,
        ):
            # Clear WEB_HOST from the environment
            os.environ.pop('WEB_HOST', None)
            mock_resend.Emails.send.return_value = mock_response

            EmailService.send_invitation_email(
                to_email='test@example.com',
                org_name='Test Org',
                inviter_name='Inviter',
                role_name='member',
                invitation_token='inv-test-token-12345',
                invitation_id=1,
            )

            call_args = mock_resend.Emails.send.call_args
            email_params = call_args[0][0]

            expected_url = f'{DEFAULT_WEB_HOST}/api/organizations/members/invite/accept?token=inv-test-token-12345'
            assert expected_url in email_params['html']


class TestEmailServiceGetResendClient:
    """Test cases for Resend client initialization."""

    def test_get_resend_client_returns_false_when_resend_not_available(self):
        """Test that _get_resend_client returns False when resend is not installed."""
        with patch('server.services.email_service.RESEND_AVAILABLE', False):
            result = EmailService._get_resend_client()
            assert result is False

    def test_get_resend_client_returns_false_when_api_key_not_configured(self):
        """Test that _get_resend_client returns False when API key is missing."""
        with (
            patch('server.services.email_service.RESEND_AVAILABLE', True),
            patch.dict(os.environ, {}, clear=True),
        ):
            os.environ.pop('RESEND_API_KEY', None)
            result = EmailService._get_resend_client()
            assert result is False

    def test_get_resend_client_returns_true_when_configured(self):
        """Test that _get_resend_client returns True when properly configured."""
        with (
            patch.dict(os.environ, {'RESEND_API_KEY': 'test-key'}),
            patch('server.services.email_service.RESEND_AVAILABLE', True),
            patch('server.services.email_service.resend') as mock_resend,
        ):
            result = EmailService._get_resend_client()
            assert result is True
            assert mock_resend.api_key == 'test-key'


class TestEmailServiceSendInvitationEmail:
    """Test cases for send_invitation_email method."""

    def test_send_invitation_email_skips_when_client_not_ready(self):
        """Test that email sending is skipped when client is not ready."""
        with patch.object(
            EmailService, '_get_resend_client', return_value=False
        ) as mock_get_client:
            # Should not raise, just return early
            EmailService.send_invitation_email(
                to_email='test@example.com',
                org_name='Test Org',
                inviter_name='Inviter',
                role_name='member',
                invitation_token='inv-test-token',
                invitation_id=1,
            )

            mock_get_client.assert_called_once()

    def test_send_invitation_email_includes_all_required_info(self):
        """Test that invitation email includes org name, inviter name, and role."""
        mock_response = MagicMock()
        mock_response.get.return_value = 'test-email-id'

        with (
            patch.dict(os.environ, {'RESEND_API_KEY': 'test-key'}),
            patch('server.services.email_service.RESEND_AVAILABLE', True),
            patch('server.services.email_service.resend') as mock_resend,
        ):
            mock_resend.Emails.send.return_value = mock_response

            EmailService.send_invitation_email(
                to_email='test@example.com',
                org_name='Acme Corp',
                inviter_name='John Doe',
                role_name='admin',
                invitation_token='inv-test-token-12345',
                invitation_id=42,
            )

            call_args = mock_resend.Emails.send.call_args
            email_params = call_args[0][0]

            # Verify email content
            assert email_params['to'] == ['test@example.com']
            assert 'Acme Corp' in email_params['subject']
            assert 'John Doe' in email_params['html']
            assert 'Acme Corp' in email_params['html']
            assert 'admin' in email_params['html']
