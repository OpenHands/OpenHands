"""Instance-level super-admin management.

Exposes a small, explicit API for managing *super admins* — users whose
``user.role_id`` references the ``admin`` role row and therefore hold the
instance-level ``superadmin`` super role (see ``server.auth.authorization``
for the super-role model).

Every endpoint here is gated by the dedicated
``Permission.MANAGE_SUPER_ADMINS`` permission, which is granted **only** to
the ``superadmin`` super role — no org-scoped role can reach these routes.
In other words: only a super admin can create or remove other super admins.

Safety invariant: the API refuses to remove the **last** remaining super
admin (enforced atomically in ``UserStore.revoke_super_admin``), so an
installation can never be locked out of instance administration. A super
admin may demote themselves as long as another super admin still exists.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator
from server.auth.authorization import Permission, require_permission
from storage.user_store import SuperAdminRevokeResult, UserStore

from openhands.app_server.utils.logger import openhands_logger as logger

super_admin_router = APIRouter(prefix='/api/admin/super-admins', tags=['Admin'])


class GrantSuperAdminRequest(BaseModel):
    """Identify the user to promote to super admin.

    Provide exactly one of ``user_id`` or ``email``. ``email`` is a
    convenience for installers/operators who key on email rather than the
    Keycloak ``sub``; it is resolved to a user id before the grant.
    """

    user_id: str | None = Field(
        default=None,
        description='Target user id (Keycloak sub / UUID string).',
    )
    email: str | None = Field(
        default=None,
        description='Target user email. Resolved to a user id before granting.',
    )

    @model_validator(mode='after')
    def _exactly_one_identifier(self) -> 'GrantSuperAdminRequest':
        if bool(self.user_id) == bool(self.email):
            raise ValueError('Provide exactly one of "user_id" or "email".')
        return self


class SuperAdminResponse(BaseModel):
    """A single super admin."""

    user_id: str
    email: str | None = None


class SuperAdminListResponse(BaseModel):
    """The full set of current super admins."""

    super_admins: list[SuperAdminResponse]


def _to_response(user) -> SuperAdminResponse:
    return SuperAdminResponse(user_id=str(user.id), email=user.email)


@super_admin_router.get('', response_model=SuperAdminListResponse)
async def list_super_admins(
    _: str = Depends(require_permission(Permission.MANAGE_SUPER_ADMINS)),
) -> SuperAdminListResponse:
    """List all current super admins. Requires ``MANAGE_SUPER_ADMINS``."""
    users = await UserStore.list_super_admins()
    return SuperAdminListResponse(super_admins=[_to_response(u) for u in users])


@super_admin_router.post(
    '',
    response_model=SuperAdminResponse,
    status_code=status.HTTP_201_CREATED,
)
async def grant_super_admin(
    body: GrantSuperAdminRequest,
    caller_user_id: str = Depends(require_permission(Permission.MANAGE_SUPER_ADMINS)),
) -> SuperAdminResponse:
    """Grant the super-admin role to an existing user.

    Idempotent: granting to a user who is already a super admin succeeds and
    returns their record. Requires ``MANAGE_SUPER_ADMINS``.
    """
    if body.email:
        target = await UserStore.get_user_by_email(body.email)
        if target is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail='No user found with that email',
            )
        target_user_id = str(target.id)
    else:
        # Validator guarantees exactly one of user_id/email is truthy, so
        # (email falsy) implies user_id is a non-empty string here. Branch on
        # truthiness -- not ``is not None`` -- to stay consistent with the
        # validator, otherwise an empty-string ``email`` would wrongly take
        # the email branch and 404.
        target_user_id = body.user_id  # type: ignore[assignment]

    try:
        user = await UserStore.grant_super_admin(target_user_id)
    except ValueError as exc:
        # Malformed user_id (not a UUID) or missing seeded admin role.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail='User not found'
        )

    logger.info(
        'super_admins:grant',
        extra={'caller_user_id': caller_user_id, 'target_user_id': target_user_id},
    )
    return _to_response(user)


@super_admin_router.delete('/{user_id}', response_model=SuperAdminResponse)
async def revoke_super_admin(
    user_id: str,
    caller_user_id: str = Depends(require_permission(Permission.MANAGE_SUPER_ADMINS)),
) -> SuperAdminResponse:
    """Revoke the super-admin role from a user (including oneself).

    Refuses with ``409 Conflict`` if the target is the only remaining super
    admin. Requires ``MANAGE_SUPER_ADMINS``.
    """
    try:
        result = await UserStore.revoke_super_admin(user_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    if result is SuperAdminRevokeResult.NOT_FOUND:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail='User not found'
        )
    if result is SuperAdminRevokeResult.NOT_SUPER_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='User is not a super admin',
        )
    if result is SuperAdminRevokeResult.LAST_SUPER_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail='Cannot remove the last remaining super admin',
        )

    logger.info(
        'super_admins:revoke',
        extra={'caller_user_id': caller_user_id, 'target_user_id': user_id},
    )
    return SuperAdminResponse(user_id=user_id)
