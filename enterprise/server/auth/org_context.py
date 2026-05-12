"""Resolve the *effective* organization ID for the current request.

Precedence (highest first):

1. ``api_key_org_id`` — org bound to the API key used for authentication.
   The key is pinned to that org and cannot be overridden. If an
   ``X-Org-Id`` header is also present and differs, the request is
   rejected with 403.

2. ``X-Org-Id`` header — explicit, per-request override sent by the
   client. Validated against the authenticated user's org memberships;
   rejected with 403 if the user is not a member of that org.

3. ``user.current_org_id`` — the user's currently selected org (as
   mutated by ``POST /api/organizations/{org_id}/switch``). Default
   fallback when neither of the above is supplied.

The resolution is cached on ``SaasUserAuth`` for the duration of a
single request so that downstream callers (route handlers, services,
SAAS conversation/pending-message injectors) all see a consistent value
without paying for the membership lookup more than once.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import Depends, Header, HTTPException, Request, status
from server.auth.saas_user_auth import SaasUserAuth
from server.logger import logger

from openhands.app_server.user_auth import get_user_auth

X_ORG_ID_HEADER = 'X-Org-Id'


async def resolve_effective_org_id(request: Request) -> UUID:
    """FastAPI dependency that returns the effective org id for this request.

    Raises:
        HTTPException 400: ``X-Org-Id`` header is present but is not a UUID.
        HTTPException 403: User is not a member of the requested org, or
            the request authenticates with an org-bound API key whose
            org does not match the ``X-Org-Id`` header.
        HTTPException 404: No effective org could be determined (e.g.
            user has no current org and did not supply the header).
    """
    user_auth = await get_user_auth(request)
    if not isinstance(user_auth, SaasUserAuth):
        # Non-SAAS deployments do not have multi-org context.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Organizations are not available in this deployment',
        )

    effective_org_id = await user_auth.get_effective_org_id()
    if effective_org_id is None:
        logger.warning(
            'effective_org_id_unavailable',
            extra={'user_id': user_auth.user_id},
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='No current organization for user',
        )
    return effective_org_id


async def maybe_resolve_effective_org_id(request: Request) -> UUID | None:
    """Variant of :func:`resolve_effective_org_id` that returns ``None``
    rather than 404 when no effective org can be determined.

    Still raises 400/403 for malformed or unauthorized ``X-Org-Id`` headers.
    """
    user_auth = await get_user_auth(request)
    if not isinstance(user_auth, SaasUserAuth):
        return None
    return await user_auth.get_effective_org_id()


async def reject_x_org_id_path_mismatch(
    request: Request,
    x_org_id: str | None = Header(default=None, alias=X_ORG_ID_HEADER),
) -> None:
    """Guard for routes with ``{org_id}`` in the path.

    These routes already pin the org via the URL: ``require_permission``
    runs against the path org, and the handlers do not consult the
    effective-org resolver. That makes any ``X-Org-Id`` header on such
    a request *redundant at best, contradictory at worst*. We silently
    accepted the conflict before, which masked client-state bugs (e.g.
    a stale org selector in the frontend sending the previous org's id
    while the user navigates to a new org's page). This dependency
    converts that into an immediate, actionable 400.

    Behavior:

    * Header absent ............................. pass
    * Header present, valid UUID, matches path .. pass
    * Header present, valid UUID, != path ....... 400
    * Header present, not a UUID ................ 400
    * Path ``org_id`` itself malformed .......... pass (FastAPI's own
      type coercion will 422 the request before the handler runs)

    Attach via ``dependencies=[REJECT_X_ORG_ID_PATH_MISMATCH]`` on each
    path-org route's decorator, or at router-level when *every* route
    on the router has ``{org_id}`` in its prefix.
    """
    if x_org_id is None:
        return

    path_org_id_raw = request.path_params.get('org_id')
    if path_org_id_raw is None:
        # Dep was attached to a route without ``{org_id}``. Treat as a
        # no-op so this can be attached at the router level to routers
        # that contain routes with and without ``{org_id}`` in the path.
        return

    try:
        header_uuid = UUID(x_org_id)
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'{X_ORG_ID_HEADER} header is not a valid UUID',
        )

    try:
        path_uuid = (
            path_org_id_raw
            if isinstance(path_org_id_raw, UUID)
            else UUID(str(path_org_id_raw))
        )
    except (ValueError, TypeError, AttributeError):
        # If the path UUID is malformed FastAPI's own coercion will
        # 422 before we reach the handler. Don't shadow that error.
        return

    if header_uuid != path_uuid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f'{X_ORG_ID_HEADER} header ({header_uuid}) does not match '
                f'the org in the request path ({path_uuid}). '
                'Remove the header or set it to the same value.'
            ),
        )


# Module-level Depends shortcuts so call sites read tidily:
#     effective_org_id: UUID = EFFECTIVE_ORG_ID,
EFFECTIVE_ORG_ID = Depends(resolve_effective_org_id)
MAYBE_EFFECTIVE_ORG_ID = Depends(maybe_resolve_effective_org_id)
REJECT_X_ORG_ID_PATH_MISMATCH = Depends(reject_x_org_id_path_mismatch)
