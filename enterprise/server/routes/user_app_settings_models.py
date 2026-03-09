"""
Pydantic models for user app settings API.
"""

from pydantic import BaseModel, EmailStr
from storage.user import User


class UserAppSettingsError(Exception):
    """Base exception for user app settings errors."""

    pass


class UserNotFoundError(UserAppSettingsError):
    """Raised when user is not found."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        super().__init__(f'User with id "{user_id}" not found')


class UserAppSettingsUpdateError(UserAppSettingsError):
    """Raised when user app settings update fails."""

    pass


class UserAppSettingsResponse(BaseModel):
    """Response model for user app settings."""

    language: str | None = None
    user_consents_to_analytics: bool | None = None
    enable_sound_notifications: bool | None = None
    git_user_name: str | None = None
    git_user_email: EmailStr | None = None

    @classmethod
    def from_user(cls, user: User) -> 'UserAppSettingsResponse':
        """Create response from User entity."""
        return cls(
            language=user.language,
            user_consents_to_analytics=user.user_consents_to_analytics,
            enable_sound_notifications=user.enable_sound_notifications,
            git_user_name=user.git_user_name,
            git_user_email=user.git_user_email,
        )


class UserAppSettingsUpdate(BaseModel):
    """Request model for updating user app settings (partial update)."""

    language: str | None = None
    user_consents_to_analytics: bool | None = None
    enable_sound_notifications: bool | None = None
    git_user_name: str | None = None
    git_user_email: EmailStr | None = None
