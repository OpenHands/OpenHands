"""SAAS-specific user models that extend OSS UserInfo with organization fields."""

from openhands.app_server.user.user_models import UserInfo


class SaasUserInfo(UserInfo):
    """User info model for SAAS mode with organization context.

    Extends the base UserInfo with SAAS-specific fields for organization
    membership, role, and permissions.
    """

    org_id: str | None = None
    org_name: str | None = None
    role: str | None = None
    permissions: list[str] | None = None
