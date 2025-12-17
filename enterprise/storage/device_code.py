"""Device code storage model for OAuth 2.0 Device Flow."""

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import Column, DateTime, Integer, String
from storage.base import Base


class DeviceCodeStatus(Enum):
    """Status of a device code authorization request."""

    PENDING = 'pending'
    AUTHORIZED = 'authorized'
    EXPIRED = 'expired'
    DENIED = 'denied'


class DeviceCode(Base):
    """Device code for OAuth 2.0 Device Flow.

    This stores the device codes issued during the device authorization flow,
    along with their status and associated user information once authorized.
    """

    __tablename__ = 'device_codes'

    id = Column(Integer, primary_key=True, autoincrement=True)
    device_code = Column(String(128), unique=True, nullable=False, index=True)
    user_code = Column(String(16), unique=True, nullable=False, index=True)
    status = Column(String(32), nullable=False, default=DeviceCodeStatus.PENDING.value)

    # Keycloak user ID who authorized the device (set during verification)
    keycloak_user_id = Column(String(255), nullable=True)

    # Timestamps
    expires_at = Column(DateTime(timezone=True), nullable=False)
    authorized_at = Column(DateTime(timezone=True), nullable=True)

    # Rate limiting fields for RFC 8628 section 3.5 compliance
    last_poll_time = Column(DateTime(timezone=True), nullable=True)
    current_interval = Column(Integer, nullable=False, default=5)

    def __repr__(self) -> str:
        return f"<DeviceCode(user_code='{self.user_code}', status='{self.status}')>"

    def is_expired(self) -> bool:
        """Check if the device code has expired."""
        now = datetime.now(timezone.utc)
        return now > self.expires_at

    def is_pending(self) -> bool:
        """Check if the device code is still pending authorization."""
        return self.status == DeviceCodeStatus.PENDING.value and not self.is_expired()

    def is_authorized(self) -> bool:
        """Check if the device code has been authorized."""
        return self.status == DeviceCodeStatus.AUTHORIZED.value

    def authorize(self, user_id: str) -> None:
        """Mark the device code as authorized."""
        self.status = DeviceCodeStatus.AUTHORIZED.value
        self.keycloak_user_id = user_id  # Set the Keycloak user ID during authorization
        self.authorized_at = datetime.now(timezone.utc)

    def deny(self) -> None:
        """Mark the device code as denied."""
        self.status = DeviceCodeStatus.DENIED.value

    def expire(self) -> None:
        """Mark the device code as expired."""
        self.status = DeviceCodeStatus.EXPIRED.value

    def check_rate_limit(self) -> tuple[bool, int]:
        """Check if the client is polling too fast.

        Returns:
            tuple: (is_too_fast, current_interval)
                - is_too_fast: True if client should receive slow_down error
                - current_interval: Current polling interval to use
        """
        now = datetime.now(timezone.utc)

        # If this is the first poll, allow it
        if self.last_poll_time is None:
            return False, self.current_interval

        # Calculate time since last poll
        time_since_last_poll = (now - self.last_poll_time).total_seconds()

        # Check if polling too fast
        if time_since_last_poll < self.current_interval:
            # Increase interval for slow_down (RFC 8628 section 3.5)
            new_interval = min(self.current_interval + 5, 60)  # Cap at 60 seconds
            return True, new_interval

        return False, self.current_interval

    def update_poll_time(self, increase_interval: bool = False) -> None:
        """Update the last poll time and optionally increase the interval.

        Args:
            increase_interval: If True, increase the current interval for slow_down
        """
        self.last_poll_time = datetime.now(timezone.utc)

        if increase_interval:
            # Increase interval by 5 seconds, cap at 60 seconds (RFC 8628)
            self.current_interval = min(self.current_interval + 5, 60)
