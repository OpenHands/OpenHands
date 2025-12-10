"""Device code storage model for OAuth 2.0 Device Flow."""

from datetime import datetime, timezone
from enum import Enum

from sqlalchemy import Column, DateTime, Integer, String, Text
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

    # User information (set when authorized - should match keycloak_user_id)
    user_id = Column(String(255), nullable=True)
    access_token = Column(Text, nullable=True)

    # Timestamps
    created_at = Column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    expires_at = Column(DateTime, nullable=False)
    authorized_at = Column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<DeviceCode(user_code='{self.user_code}', status='{self.status}')>"

    def is_expired(self) -> bool:
        """Check if the device code has expired."""
        now = datetime.now(timezone.utc)
        expires_at = self.expires_at

        # Handle timezone-naive datetime from database
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        return now > expires_at

    def is_pending(self) -> bool:
        """Check if the device code is still pending authorization."""
        return self.status == DeviceCodeStatus.PENDING.value and not self.is_expired()

    def is_authorized(self) -> bool:
        """Check if the device code has been authorized."""
        return self.status == DeviceCodeStatus.AUTHORIZED.value

    def authorize(self, user_id: str, access_token: str) -> None:
        """Mark the device code as authorized with user API key."""
        self.status = DeviceCodeStatus.AUTHORIZED.value
        self.keycloak_user_id = user_id  # Set the Keycloak user ID during authorization
        self.user_id = user_id
        self.access_token = access_token
        self.authorized_at = datetime.now(timezone.utc)

    def deny(self) -> None:
        """Mark the device code as denied."""
        self.status = DeviceCodeStatus.DENIED.value

    def expire(self) -> None:
        """Mark the device code as expired."""
        self.status = DeviceCodeStatus.EXPIRED.value
