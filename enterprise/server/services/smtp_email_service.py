"""Email service for sending transactional emails via SMTP."""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import TypedDict

from openhands.app_server.utils.logger import openhands_logger as logger

DEFAULT_FROM_EMAIL = 'OpenHands <no-reply@openhands.dev>'
DEFAULT_WEB_HOST = 'https://app.all-hands.dev'


class SMTPSettings(TypedDict):
    host: str
    port: int
    use_ssl: bool
    use_tls: bool
    username: str
    password: str
    from_email: str


class SMTPEmailService:
    """Service for sending transactional emails via SMTP."""

    @staticmethod
    def _get_smtp_settings() -> SMTPSettings | None:
        host = os.environ.get('SMTP_HOST')
        if not host:
            return None

        port_value = os.environ.get('SMTP_PORT', '587')
        try:
            port = int(port_value)
        except ValueError:
            logger.warning(
                'SMTP_PORT invalid, defaulting to 587',
                extra={'smtp_port': port_value},
            )
            port = 587

        use_ssl = os.environ.get('SMTP_USE_SSL', 'false').lower() in (
            'true',
            '1',
        )
        use_tls = (
            os.environ.get('SMTP_USE_TLS', 'true').lower() in ('true', '1')
            and not use_ssl
        )

        return {
            'host': host,
            'port': port,
            'use_ssl': use_ssl,
            'use_tls': use_tls,
            'username': os.environ.get('SMTP_USERNAME', ''),
            'password': os.environ.get('SMTP_PASSWORD', ''),
            'from_email': os.environ.get('SMTP_FROM_EMAIL', DEFAULT_FROM_EMAIL),
        }

    @staticmethod
    def _send_smtp_email(
        to_emails: list[str],
        subject: str,
        html: str,
        extra: dict[str, object] | None = None,
    ) -> bool:
        settings = SMTPEmailService._get_smtp_settings()
        if not settings:
            logger.warning('SMTP not configured, skipping email')
            return False

        message = MIMEMultipart('alternative')
        message['Subject'] = subject
        message['From'] = str(settings['from_email'])
        message['To'] = ', '.join(to_emails)
        message.attach(MIMEText(html, 'html'))

        extra_payload = extra or {}

        try:
            smtp_class = smtplib.SMTP_SSL if settings['use_ssl'] else smtplib.SMTP
            client = smtp_class(settings['host'], settings['port'])
            try:
                if settings['use_tls']:
                    client.starttls()
                if settings['username']:
                    client.login(settings['username'], settings['password'])
                client.sendmail(
                    settings['from_email'],
                    to_emails,
                    message.as_string(),
                )
            finally:
                client.quit()
            return True
        except Exception:
            logger.exception(
                'Failed to send SMTP email', extra={**extra_payload}, stack_info=True
            )
            return False

    @staticmethod
    def is_configured() -> bool:
        """Whether transactional email delivery is configured.

        Returns True when SMTP settings are available so callers can surface
        "email is not configured" to users instead of letting invitations fail
        silently.
        """
        return SMTPEmailService._get_smtp_settings() is not None

    @staticmethod
    def build_invitation_url(invitation_token: str) -> str:
        """Build the absolute acceptance URL for an invitation token.

        WEB_HOST may be configured as a bare hostname (the OHE chart sets it
        that way), so normalize to an https URL before composing the link.
        """
        web_host = os.environ.get('WEB_HOST', DEFAULT_WEB_HOST).strip().rstrip('/')
        if not web_host:
            web_host = DEFAULT_WEB_HOST
        if not web_host.startswith(('http://', 'https://')):
            web_host = f'https://{web_host}'
        return (
            f'{web_host}/api/organizations/members/invite/accept'
            f'?token={invitation_token}'
        )

    @staticmethod
    def send_invitation_email(
        to_email: str,
        org_name: str,
        inviter_name: str,
        role_name: str,
        invitation_token: str,
        invitation_id: int,
    ) -> None:
        """Send an organization invitation email.

        Args:
            to_email: Recipient's email address
            org_name: Name of the organization
            inviter_name: Display name of the person who sent the invite
            role_name: Role being offered (e.g., 'member', 'admin')
            invitation_token: The secure invitation token
            invitation_id: The invitation ID for logging
        """
        invitation_url = SMTPEmailService.build_invitation_url(invitation_token)
        subject = f"You're invited to join {org_name} on OpenHands"
        body = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <p>Hi,</p>

            <p><strong>{inviter_name}</strong> has invited you to join <strong>{org_name}</strong> on OpenHands as a <strong>{role_name}</strong>.</p>

            <p>Click the button below to accept the invitation:</p>

            <p style="margin: 30px 0;">
                <a href="{invitation_url}"
                   style="background-color: #c9b974; color: #0D0F11; padding: 8px 16px;
                          text-decoration: none; border-radius: 8px; display: inline-block;
                          font-size: 14px; font-weight: 600;">
                    Accept Invitation
                </a>
            </p>

            <p style="color: #666; font-size: 14px;">
                Or copy and paste this link into your browser:<br>
                <a href="{invitation_url}" style="color: #c9b974; font-weight: 600;">{invitation_url}</a>
            </p>

            <p style="color: #666; font-size: 14px;">
                This invitation will expire in 7 days.
            </p>

            <p style="color: #666; font-size: 14px;">
                If you weren't expecting this invitation, you can safely ignore this email.
            </p>

            <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">

            <p style="color: #999; font-size: 12px;">
                Best,<br>
                The OpenHands Team
            </p>
        </div>
        """

        extra = {'invitation_id': invitation_id, 'email': to_email}

        if SMTPEmailService._send_smtp_email([to_email], subject, body, extra):
            logger.info('Invitation email sent', extra=extra)

    @staticmethod
    def send_budget_alert_email(
        to_emails: list[str],
        org_name: str,
        percentage: float,
        current_spend: float,
        monthly_limit: float,
        threshold: int,
    ) -> None:
        subject = (
            f'OpenHands budget alert: {org_name} reached {threshold}% of its limit'
        )
        body = f"""
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <p>Hi,</p>
            <p><strong>{org_name}</strong> has reached <strong>{percentage:.1f}%</strong>
            of its monthly budget.</p>
            <p><strong>Current spend:</strong> ${current_spend:,.2f}<br />
            <strong>Monthly limit:</strong> ${monthly_limit:,.2f}<br />
            <strong>Threshold:</strong> {threshold}%</p>
            <p>Please review your Usage & Monitoring dashboard for more details.</p>
            <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">
            <p style="color: #999; font-size: 12px;">Best,<br>The OpenHands Team</p>
        </div>
        """

        extra = {'org_name': org_name, 'recipient_count': len(to_emails)}

        if SMTPEmailService._send_smtp_email(to_emails, subject, body, extra):
            logger.info('Budget alert email sent via SMTP', extra=extra)
