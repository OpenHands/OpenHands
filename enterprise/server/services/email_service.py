"""Email service for sending transactional emails via Resend."""

import os

try:
    import resend

    RESEND_AVAILABLE = True
except ImportError:
    RESEND_AVAILABLE = False

from openhands.core.logger import openhands_logger as logger

DEFAULT_FROM_EMAIL = 'OpenHands <no-reply@openhands.dev>'
DEFAULT_WEB_HOST = 'https://app.all-hands.dev'


class EmailService:
    """Service for sending transactional emails."""

    @staticmethod
    def _get_resend_client() -> bool:
        """Initialize and return the Resend client.

        Returns:
            bool: True if client is ready, False otherwise
        """
        if not RESEND_AVAILABLE:
            logger.warning('Resend library not installed, skipping email')
            return False

        resend_api_key = os.environ.get('RESEND_API_KEY')
        if not resend_api_key:
            logger.warning('RESEND_API_KEY not configured, skipping email')
            return False

        resend.api_key = resend_api_key
        return True

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
        if not EmailService._get_resend_client():
            return

        # Build invitation URL
        web_host = os.environ.get('WEB_HOST', DEFAULT_WEB_HOST)
        invitation_url = f'{web_host}/api/organizations/members/invite/accept?token={invitation_token}'

        from_email = os.environ.get('RESEND_FROM_EMAIL', DEFAULT_FROM_EMAIL)

        params = {
            'from': from_email,
            'to': [to_email],
            'subject': f"You're invited to join {org_name} on OpenHands",
            'html': f"""
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
            """,
        }

        try:
            response = resend.Emails.send(params)
            logger.info(
                'Invitation email sent',
                extra={
                    'invitation_id': invitation_id,
                    'email': to_email,
                    'response_id': response.get('id') if response else None,
                },
            )
        except Exception as e:
            logger.error(
                'Failed to send invitation email',
                extra={
                    'invitation_id': invitation_id,
                    'email': to_email,
                    'error': str(e),
                },
            )
            raise
