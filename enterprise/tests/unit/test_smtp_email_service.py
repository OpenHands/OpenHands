"""Tests for SMTP email service."""

import os
from unittest.mock import MagicMock, patch

from server.services.smtp_email_service import (
    DEFAULT_WEB_HOST,
    SMTPEmailService,
)


class TestSMTPEmailServiceInvitationUrl:
    """Test cases for invitation URL generation."""

    def test_invitation_url_uses_correct_endpoint(self):
        """Test that invitation URL points to the correct API endpoint."""
        smtp_client = MagicMock()

        with (
            patch.dict(os.environ, {'SMTP_HOST': 'smtp.example.com'}, clear=True),
            patch(
                'server.services.smtp_email_service.smtplib.SMTP',
                return_value=smtp_client,
            ),
        ):
            SMTPEmailService.send_invitation_email(
                to_email='test@example.com',
                org_name='Test Org',
                inviter_name='Inviter',
                role_name='member',
                invitation_token='inv-test-token-12345',
                invitation_id=1,
            )

            message = smtp_client.sendmail.call_args[0][2]
            assert '/api/organizations/members/invite/accept?token=' in message
            assert 'inv-test-token-12345' in message

    def test_invitation_url_uses_web_host_env_var(self):
        """Test that invitation URL uses WEB_HOST environment variable."""
        custom_host = 'https://custom.example.com'
        smtp_client = MagicMock()

        with (
            patch.dict(
                os.environ,
                {'SMTP_HOST': 'smtp.example.com', 'WEB_HOST': custom_host},
                clear=True,
            ),
            patch(
                'server.services.smtp_email_service.smtplib.SMTP',
                return_value=smtp_client,
            ),
        ):
            SMTPEmailService.send_invitation_email(
                to_email='test@example.com',
                org_name='Test Org',
                inviter_name='Inviter',
                role_name='member',
                invitation_token='inv-test-token-12345',
                invitation_id=1,
            )

            message = smtp_client.sendmail.call_args[0][2]
            expected_url = f'{custom_host}/api/organizations/members/invite/accept?token=inv-test-token-12345'
            assert expected_url in message

    def test_invitation_url_uses_default_host_when_env_not_set(self):
        """Test that invitation URL falls back to DEFAULT_WEB_HOST when env not set."""
        smtp_client = MagicMock()

        with (
            patch.dict(os.environ, {'SMTP_HOST': 'smtp.example.com'}, clear=True),
            patch(
                'server.services.smtp_email_service.smtplib.SMTP',
                return_value=smtp_client,
            ),
        ):
            SMTPEmailService.send_invitation_email(
                to_email='test@example.com',
                org_name='Test Org',
                inviter_name='Inviter',
                role_name='member',
                invitation_token='inv-test-token-12345',
                invitation_id=1,
            )

            message = smtp_client.sendmail.call_args[0][2]
            expected_url = f'{DEFAULT_WEB_HOST}/api/organizations/members/invite/accept?token=inv-test-token-12345'
            assert expected_url in message


class TestSMTPEmailServiceSendInvitationEmail:
    """Test cases for send_invitation_email method."""

    def test_send_invitation_email_skips_when_smtp_not_configured(self):
        """Test that email sending is skipped when SMTP is not configured."""
        with patch.object(
            SMTPEmailService, '_send_smtp_email', return_value=False
        ) as mock_send:
            SMTPEmailService.send_invitation_email(
                to_email='test@example.com',
                org_name='Test Org',
                inviter_name='Inviter',
                role_name='member',
                invitation_token='inv-test-token',
                invitation_id=1,
            )

            mock_send.assert_called_once()

    def test_send_invitation_email_includes_all_required_info(self):
        """Test that invitation email includes org name, inviter name, and role."""
        smtp_client = MagicMock()

        with (
            patch.dict(
                os.environ,
                {
                    'SMTP_HOST': 'smtp.example.com',
                    'SMTP_FROM_EMAIL': 'alerts@example.com',
                },
                clear=True,
            ),
            patch(
                'server.services.smtp_email_service.smtplib.SMTP',
                return_value=smtp_client,
            ),
        ):
            SMTPEmailService.send_invitation_email(
                to_email='test@example.com',
                org_name='Acme Corp',
                inviter_name='John Doe',
                role_name='admin',
                invitation_token='inv-test-token-12345',
                invitation_id=42,
            )

            message = smtp_client.sendmail.call_args[0][2]

            assert 'Acme Corp' in message
            assert 'John Doe' in message
            assert 'admin' in message
            assert "You're invited to join Acme Corp on OpenHands" in message


class TestSMTPEmailServiceBudgetAlerts:
    """Test cases for budget alert emails."""

    def test_send_budget_alert_uses_smtp_when_configured(self):
        smtp_client = MagicMock()

        with (
            patch.dict(
                os.environ,
                {
                    'SMTP_HOST': 'smtp.example.com',
                    'SMTP_PORT': '2525',
                    'SMTP_FROM_EMAIL': 'alerts@example.com',
                },
                clear=True,
            ),
            patch(
                'server.services.smtp_email_service.smtplib.SMTP',
                return_value=smtp_client,
            ) as mock_smtp,
        ):
            SMTPEmailService.send_budget_alert_email(
                to_emails=['ops@example.com'],
                org_name='Acme Corp',
                percentage=80.0,
                current_spend=800.0,
                monthly_limit=1000.0,
                threshold=80,
            )

            mock_smtp.assert_called_once_with('smtp.example.com', 2525)
            smtp_client.starttls.assert_called_once()
            smtp_client.sendmail.assert_called_once()
            send_args = smtp_client.sendmail.call_args[0]
            assert send_args[0] == 'alerts@example.com'
            assert send_args[1] == ['ops@example.com']

            smtp_client.quit.assert_called_once()


class TestSMTPEmailServiceHelpers:
    """Tests for is_configured and build_invitation_url."""

    def test_is_configured_false_without_smtp_host(self, monkeypatch):
        monkeypatch.delenv('SMTP_HOST', raising=False)
        from server.services.smtp_email_service import SMTPEmailService

        assert SMTPEmailService.is_configured() is False

    def test_is_configured_true_with_smtp_host(self, monkeypatch):
        monkeypatch.setenv('SMTP_HOST', 'smtp.example.com')
        from server.services.smtp_email_service import SMTPEmailService

        assert SMTPEmailService.is_configured() is True

    def test_build_invitation_url_normalizes_bare_hostname(self, monkeypatch):
        """OHE charts set WEB_HOST as a bare hostname; links must get a scheme."""
        monkeypatch.setenv('WEB_HOST', 'app.example.com')
        from server.services.smtp_email_service import SMTPEmailService

        url = SMTPEmailService.build_invitation_url('inv-token123')

        assert url == (
            'https://app.example.com/api/organizations/members/invite/accept'
            '?token=inv-token123'
        )

    def test_build_invitation_url_keeps_explicit_scheme(self, monkeypatch):
        monkeypatch.setenv('WEB_HOST', 'https://app.example.com/')
        from server.services.smtp_email_service import SMTPEmailService

        url = SMTPEmailService.build_invitation_url('inv-token123')

        assert url.startswith('https://app.example.com/api/')
