"""Device code store for OAuth 2.0 Device Flow."""

import secrets
import string
from datetime import datetime, timedelta, timezone

from storage.device_code import DeviceCode


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

    def _generate_unique_codes(
        self, session, max_attempts: int = 10
    ) -> tuple[str, str]:
        """Generate unique user and device codes.

        Args:
            session: Database session
            max_attempts: Maximum number of attempts to generate unique codes

        Returns:
            Tuple of (user_code, device_code)

        Raises:
            RuntimeError: If unable to generate unique codes after max_attempts
        """
        for _ in range(max_attempts):
            user_code = self.generate_user_code()
            device_code = self.generate_device_code()

            # Check uniqueness with a single query using OR condition
            existing = (
                session.query(DeviceCode)
                .filter(
                    (DeviceCode.user_code == user_code)
                    | (DeviceCode.device_code == device_code)
                )
                .first()
            )

            if not existing:
                return user_code, device_code

        raise RuntimeError(
            f'Failed to generate unique device codes after {max_attempts} attempts'
        )

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
            user_code, device_code = self._generate_unique_codes(session)
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

    def get_by_device_code(self, device_code: str) -> DeviceCode | None:
        """Get device code entry by device code."""
        with self.session_maker() as session:
            result = (
                session.query(DeviceCode).filter_by(device_code=device_code).first()
            )
            if result:
                session.expunge(result)  # Detach from session cleanly
            return result

    def get_by_user_code(self, user_code: str) -> DeviceCode | None:
        """Get device code entry by user code."""
        with self.session_maker() as session:
            result = session.query(DeviceCode).filter_by(user_code=user_code).first()
            if result:
                session.expunge(result)  # Detach from session cleanly
            return result

    def authorize_device_code(self, user_code: str, user_id: str) -> bool:
        """Authorize a device code.

        Args:
            user_code: The user code to authorize
            user_id: The user ID from Keycloak

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

            device_code_entry.authorize(user_id)
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
