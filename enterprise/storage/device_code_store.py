"""Device code store for OAuth 2.0 Device Flow."""

import secrets
import string
from datetime import datetime, timedelta, timezone
from typing import Optional

from storage.device_code import DeviceCode, DeviceCodeStatus


class DeviceCodeStore:
    """Store for managing OAuth 2.0 device codes."""

    def __init__(self, session_maker):
        self.session_maker = session_maker

    def generate_user_code(self) -> str:
        """Generate a human-readable user code (8 characters, uppercase letters and digits)."""
        # Use a mix of uppercase letters and digits, avoiding confusing characters
        alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'  # No I, O, 0, 1
        return ''.join(secrets.choice(alphabet) for _ in range(8))

    def generate_device_code(self) -> str:
        """Generate a secure device code (128 characters)."""
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(128))

    def create_device_code(
        self,
        expires_in: int = 600,  # 10 minutes default
    ) -> DeviceCode:
        """Create a new device code entry.

        Args:
            expires_in: Expiration time in seconds

        Returns:
            The created DeviceCode instance
        """
        with self.session_maker() as session:
            # Generate unique codes
            max_attempts = 10
            for _ in range(max_attempts):
                user_code = self.generate_user_code()
                device_code = self.generate_device_code()

                # Check if codes are unique
                existing_user = (
                    session.query(DeviceCode).filter_by(user_code=user_code).first()
                )
                existing_device = (
                    session.query(DeviceCode).filter_by(device_code=device_code).first()
                )

                if not existing_user and not existing_device:
                    break
            else:
                raise RuntimeError(
                    'Failed to generate unique device codes after multiple attempts'
                )

            expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)

            device_code_entry = DeviceCode(
                device_code=device_code,
                user_code=user_code,
                keycloak_user_id=None,  # Will be set during authorization
                expires_at=expires_at,
            )

            session.add(device_code_entry)
            session.commit()
            session.refresh(device_code_entry)

            return device_code_entry

    def get_by_device_code(self, device_code: str) -> Optional[DeviceCode]:
        """Get device code entry by device code."""
        with self.session_maker() as session:
            return session.query(DeviceCode).filter_by(device_code=device_code).first()

    def get_by_user_code(self, user_code: str) -> Optional[DeviceCode]:
        """Get device code entry by user code."""
        with self.session_maker() as session:
            return session.query(DeviceCode).filter_by(user_code=user_code).first()

    def authorize_device_code(self, user_code: str, user_id: str, api_key: str) -> bool:
        """Authorize a device code with user's API key.

        Args:
            user_code: The user code to authorize
            user_id: The user ID from Keycloak
            api_key: The user's API key

        Returns:
            True if authorization was successful, False otherwise
        """
        with self.session_maker() as session:
            device_code_entry = (
                session.query(DeviceCode).filter_by(user_code=user_code).first()
            )

            if not device_code_entry:
                return False

            if not device_code_entry.is_pending():
                return False

            device_code_entry.authorize(user_id, api_key)
            session.commit()

            return True

    def deny_device_code(self, user_code: str) -> bool:
        """Deny a device code authorization.

        Args:
            user_code: The user code to deny

        Returns:
            True if denial was successful, False otherwise
        """
        with self.session_maker() as session:
            device_code_entry = (
                session.query(DeviceCode).filter_by(user_code=user_code).first()
            )

            if not device_code_entry:
                return False

            if not device_code_entry.is_pending():
                return False

            device_code_entry.deny()
            session.commit()

            return True

    def cleanup_expired_codes(self) -> int:
        """Clean up expired device codes.

        Returns:
            Number of expired codes cleaned up
        """
        with self.session_maker() as session:
            expired_codes = (
                session.query(DeviceCode)
                .filter(
                    DeviceCode.expires_at < datetime.now(timezone.utc),
                    DeviceCode.status == DeviceCodeStatus.PENDING.value,
                )
                .all()
            )

            count = 0
            for code in expired_codes:
                code.expire()
                count += 1

            session.commit()
            return count
