"""
Pydantic models and custom exceptions for organization invitations.
"""

from pydantic import BaseModel, EmailStr
from storage.org_invitation import OrgInvitation
from storage.role_store import RoleStore


class InvitationError(Exception):
    """Base exception for invitation errors."""

    pass


class InvitationAlreadyExistsError(InvitationError):
    """Raised when a pending invitation already exists for the email."""

    def __init__(
        self, message: str = 'A pending invitation already exists for this email'
    ):
        super().__init__(message)


class UserAlreadyMemberError(InvitationError):
    """Raised when the user is already a member of the organization."""

    def __init__(self, message: str = 'User is already a member of this organization'):
        super().__init__(message)


class InvitationExpiredError(InvitationError):
    """Raised when the invitation has expired."""

    def __init__(self, message: str = 'Invitation has expired'):
        super().__init__(message)


class InvitationInvalidError(InvitationError):
    """Raised when the invitation is invalid or revoked."""

    def __init__(self, message: str = 'Invitation is no longer valid'):
        super().__init__(message)


class InsufficientPermissionError(InvitationError):
    """Raised when the user lacks permission to perform the action."""

    def __init__(self, message: str = 'Insufficient permission'):
        super().__init__(message)


class EmailMismatchError(InvitationError):
    """Raised when the accepting user's email doesn't match the invitation email."""

    def __init__(self, message: str = 'Your email does not match the invitation'):
        super().__init__(message)


class InvitationCreate(BaseModel):
    """Request model for creating invitation(s)."""

    emails: list[EmailStr]
    role: str = 'member'  # Default to member role


class InvitationResponse(BaseModel):
    """Response model for invitation details."""

    id: int
    email: str
    role: str
    status: str
    created_at: str
    expires_at: str
    inviter_email: str | None = None

    @classmethod
    def from_invitation(
        cls,
        invitation: OrgInvitation,
        inviter_email: str | None = None,
    ) -> 'InvitationResponse':
        """Create an InvitationResponse from an OrgInvitation entity.

        Args:
            invitation: The invitation entity to convert
            inviter_email: Optional email of the inviter

        Returns:
            InvitationResponse: The response model instance
        """
        role_name = ''
        if invitation.role:
            role_name = invitation.role.name
        elif invitation.role_id:
            role = RoleStore.get_role_by_id(invitation.role_id)
            role_name = role.name if role else ''

        return cls(
            id=invitation.id,
            email=invitation.email,
            role=role_name,
            status=invitation.status,
            created_at=invitation.created_at.isoformat(),
            expires_at=invitation.expires_at.isoformat(),
            inviter_email=inviter_email,
        )


class InvitationFailure(BaseModel):
    """Response model for a failed invitation."""

    email: str
    error: str


class BatchInvitationResponse(BaseModel):
    """Response model for batch invitation creation."""

    successful: list[InvitationResponse]
    failed: list[InvitationFailure]
