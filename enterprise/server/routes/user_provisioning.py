"""
Admin endpoint for provisioning new users directly into an organization.

This is a privileged operation: it bypasses the normal sign-up flow
(email verification, TOS acceptance, OAuth IDP round-trip) and creates a
ready-to-use account on behalf of an org admin. Access is gated by the
``PROVISION_USER`` permission, which is granted to org-scoped ``owner``
and ``admin`` roles and explicit instance-level super roles (see
``server.auth.authorization``).

Flow (POST ``/api/organizations/provision-user``):

1. Resolve the target org from the ``X-Org-Id`` header (validated against
   the caller's memberships by ``require_permission``).
2. Create the user in Keycloak (realm configured by
   ``KEYCLOAK_REALM_NAME``), pre-verifying the email and setting a
   non-temporary password.
3. Request an offline refresh token for the new user via ROPC and store
   it via ``TokenManager.store_offline_token`` so the account can be used
   for backend operations (e.g. SDK calls) immediately.
4. Create the OpenHands user record (mirroring the ``keycloak_callback``
   shape), but with ``email_verified=True``, ``accepted_tos=now()``, and
   ``user_consents_to_analytics=True`` — no UI round-trips required.
5. Set up a LiteLLM integration for the user in the target org and add
   them with the requested role (``member`` by default, ``admin``, or ``owner``).
6. Mint an API key bound to the target org and return it to the caller.

The caller receives the email, password (generated if not supplied) and
API key in a single response. On failure after the Keycloak user is
created, ``_rollback_partial_provision`` compensates whatever subset of
steps already ran — see its docstring for the full unwind order.

**Compensation strategy.** Each post-Keycloak side effect is tracked by
a local progress variable so the unwind path can target exactly the
state that was created:

* ``openhands_user_id`` — set once ``UserStore.create_user`` returns,
  marking that the User row, personal Org, owner OrgMember row, default
  settings, and personal-org LiteLLM team all exist.
* ``target_membership_added`` — set once
  ``OrgMemberStore.add_user_to_org`` succeeds for the target org.

The order in the unwind matters: the personal-org cascade only
deletes the User row when the user is the sole orphan of the personal
org, so the target-org membership (if added) must be removed *before*
the cascade runs. See ``_rollback_partial_provision`` for the full
ordering and the rationale behind it.

**Known partial-cleanup gap (offline token).** Step 3a stores the
offline token before ``UserStore.create_user`` runs, mirroring the
production OAuth flow (``keycloak_offline_callback`` stores the
token before any ``UserStore`` interaction). The rollback path
removes the Keycloak user, which makes the orphaned offline token
row harmless — it is keyed by a Keycloak ``sub`` that no longer
exists, so it cannot be used to authenticate. The inverse ordering
(user-first) would leak the *entire* OpenHands cascade instead of a
single encrypted token blob, which is strictly worse. Periodic
``OfflineTokenStore`` reconciliation (if/when added) can sweep these.

**Known partial-cleanup gap (LiteLLM target-team membership).** When
``create_litellm_integration`` succeeded for the target org, the
provisioned Keycloak ``sub`` was added to the target org's LiteLLM
team and a per-user key was minted on the LiteLLM side. Removing the
OpenHands ``OrgMember`` row does not propagate to LiteLLM, so on
rollback the LiteLLM-side membership and key for that ``sub`` are
left behind. They are functionally inert (the Keycloak account is
deleted in the same unwind, so there is no way to authenticate as
the orphaned ``sub``), but they do accumulate. Cleanup should ride
on a future ``LiteLlmManager.remove_user_from_team(sub, org_id)``
helper rather than reaching into private internals from this route.
"""

from __future__ import annotations

import secrets
import string
from datetime import datetime, timezone
from typing import Literal
from uuid import UUID
from uuid import UUID as parse_uuid

from fastapi import APIRouter, Depends, HTTPException, status
from keycloak.exceptions import KeycloakError
from pydantic import BaseModel, EmailStr, Field, SecretStr
from server.auth.authorization import Permission, require_permission
from server.auth.org_context import EFFECTIVE_ORG_ID
from server.auth.token_manager import TokenManager
from sqlalchemy import select
from storage.api_key_store import ApiKeyStore
from storage.database import a_session_maker
from storage.org_member_store import OrgMemberStore
from storage.org_service import OrgService
from storage.org_store import OrgStore
from storage.role_store import RoleStore
from storage.user import User
from storage.user_store import UserStore

from openhands.app_server.utils.logger import openhands_logger as logger

# Routes that read the target org from ``X-Org-Id`` rather than the URL
# path live under ``/api/organizations`` so they sit alongside the rest
# of the org-management surface, but they are intentionally separated
# from ``org_router`` (which carries the ``REJECT_X_ORG_ID_PATH_MISMATCH``
# guard for ``/{org_id}/...`` routes — that guard would no-op here, but
# keeping the routers split makes the intent explicit).
user_provisioning_router = APIRouter(prefix='/api/organizations', tags=['Orgs'])

# Roles that can be assigned directly during provisioning. Provisioning
# supports creating regular members, org admins, and org owners.
ProvisionedRoleName = Literal['member', 'admin', 'owner']
DEFAULT_PROVISIONED_ROLE: ProvisionedRoleName = 'member'

# Length of generated passwords. 24 characters from a 70-symbol alphabet
# yields well over 128 bits of entropy, which exceeds typical Keycloak
# realm password-policy minimums while staying short enough to display
# in API responses.
_GENERATED_PASSWORD_LENGTH = 24


def _utc_now_naive() -> datetime:
    """Return the current UTC time as a naive ``datetime``.

    ``User.accepted_tos`` (and similar timestamp columns on the user
    record) are stored as naive UTC datetimes, so we strip the tzinfo
    after capturing ``now`` in UTC to avoid mixed-awareness comparisons.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _generate_password(length: int = _GENERATED_PASSWORD_LENGTH) -> str:
    """Generate a strong random password suitable for Keycloak policies.

    Mixes upper/lowercase letters, digits, and a curated set of symbols
    so the result satisfies common Keycloak password-policy rules
    (digits, special characters, mixed case) without including
    characters that complicate shell/JSON usage.
    """
    alphabet = string.ascii_letters + string.digits + '!@#$%^&*-_=+'
    # Loop until we have at least one of each character class to
    # satisfy realm password policies that mandate mixed character
    # types. With a 24-character draw from this alphabet the
    # probability of any class being absent is ~10^-9, so a bounded
    # loop guarantees termination while still effectively always
    # succeeding on the first iteration.
    for _ in range(100):
        candidate = ''.join(secrets.choice(alphabet) for _ in range(length))
        if (
            any(c.islower() for c in candidate)
            and any(c.isupper() for c in candidate)
            and any(c.isdigit() for c in candidate)
            and any(c in '!@#$%^&*-_=+' for c in candidate)
        ):
            return candidate
    # Practically unreachable: after 100 independent 24-char draws the
    # probability of never satisfying the class constraints is < 10^-87.
    raise RuntimeError(
        'Failed to generate a password satisfying character-class '
        'requirements after 100 attempts.'
    )


class ProvisionUserRequest(BaseModel):
    """Payload for ``POST /api/organizations/provision-user``."""

    email: EmailStr = Field(
        ...,
        description='Email address for the new user. Used as the Keycloak username.',
    )
    password: str | None = Field(
        default=None,
        min_length=8,
        max_length=256,
        description=(
            'Optional initial password. If omitted, a strong random '
            'password is generated and returned in the response. '
            'When supplied, this value is sent to Keycloak as-is and '
            'must satisfy the realm password policy (length, character '
            'classes, blacklist, etc.); a policy violation surfaces as '
            'a 409 from Keycloak. Generated passwords are constructed '
            'to satisfy common realm policies (mixed case + digit + '
            'symbol) — caller-supplied passwords carry no such '
            'guarantees beyond the ``min_length=8`` floor enforced here.'
        ),
    )
    api_key_name: str | None = Field(
        default=None,
        max_length=255,
        description='Optional name for the generated API key.',
    )
    role: ProvisionedRoleName = Field(
        default=DEFAULT_PROVISIONED_ROLE,
        description=(
            'Role to assign in the target organization. Provisioning supports '
            'member, admin, and owner.'
        ),
    )


class ProvisionUserResponse(BaseModel):
    """Response for ``POST /api/organizations/provision-user``.

    ``password`` is **intentionally** returned to the caller — this is
    the *only* time the plaintext is available, because the endpoint
    bypasses the normal email-based set-password flow. The admin who
    called this endpoint is expected to hand the credential to the new
    user out-of-band (e.g. an internal IT system, secrets manager, or
    direct hand-off). Callers should treat the response body as
    sensitive: do not log it, and prefer TLS-terminated transport.
    """

    email: str
    password: str = Field(
        description=(
            'Plaintext initial password for the new user. Either the '
            "caller's supplied value or a freshly generated one. "
            'Returned so the admin can transmit it out-of-band; this '
            'is the only point at which it is recoverable.'
        ),
    )
    api_key: str
    user_id: str
    org_id: str
    role: str


async def _set_user_provisioned_flags(user_id: str) -> None:
    """Stamp ``email_verified``, ``accepted_tos`` and analytics consent.

    ``UserStore.create_user`` already wires up the user, personal org,
    and org-member rows. The provisioning flow then bypasses the UI
    onboarding interstitials by stamping the flags directly so the
    provisioned account is fully usable immediately. Kept as a focused
    helper to keep the route handler readable.
    """
    async with a_session_maker() as session:
        result = await session.execute(
            select(User).where(User.id == parse_uuid(user_id))
        )
        user = result.scalar_one_or_none()
        if not user:
            return
        user.email_verified = True
        user.accepted_tos = _utc_now_naive()
        user.user_consents_to_analytics = True
        # Provisioned users skip the in-product onboarding form; the
        # admin has already onboarded them out-of-band.
        user.onboarding_completed = True
        await session.commit()


@user_provisioning_router.post(
    '/provision-user',
    response_model=ProvisionUserResponse,
    status_code=status.HTTP_201_CREATED,
)
async def provision_user(
    body: ProvisionUserRequest,
    caller_user_id: str = Depends(require_permission(Permission.PROVISION_USER)),
    target_org_id: UUID = EFFECTIVE_ORG_ID,
) -> ProvisionUserResponse:
    """Create a new user and add them to the caller's selected org.

    The target org is the API key org if an API key is used, otherwise it is
    taken from the ``X-Org-Id`` header (resolved by ``EFFECTIVE_ORG_ID``).
    The caller must hold the ``PROVISION_USER`` permission for that org.
    Org-scoped owners/admins have it, and super roles may grant it explicitly
    without org membership.

    Returns the email, password (generated if not supplied) and the
    new user's API key bound to the target org.
    """
    email = body.email.lower().strip()
    password = body.password or _generate_password()
    api_key_name = body.api_key_name or 'Initial API Key'
    provisioned_role = body.role

    # Confirm the target org actually exists before we mutate Keycloak.
    # ``require_permission`` has already validated that the caller is a
    # member with the right role, but the org row could have been
    # deleted between the membership check and this code path; an
    # explicit fetch produces a clean 404 instead of a confusing 500
    # later in ``OrgService.create_litellm_integration``.
    org = await OrgStore.get_org_by_id(target_org_id)
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='Target organization not found',
        )

    # Reject provisioning into a personal workspace.
    #
    # The personal-workspace invariant is ``Org.id == User.id == UUID(
    # keycloak.sub)`` (see ``UserStore.create_user``), so a personal
    # workspace is detected by comparing ``target_org_id`` to the
    # *caller's* user id. ``require_permission(PROVISION_USER)`` lets
    # the call through because every user is the owner of their own
    # personal org, but the *product* meaning of the permission is
    # "create members of a team org" — not "every user can mint extra
    # accounts in their personal workspace and receive credentials/
    # API keys for them". Mirrors the long-standing personal-workspace
    # rejection in ``server.services.org_invitation_service`` (and uses
    # the same 403 status code) so the two endpoints behave
    # consistently.
    if str(target_org_id) == caller_user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail='Cannot provision users into a personal workspace',
        )

    token_manager = TokenManager()

    # 1. Create the Keycloak user. Any failure here is terminal — we
    # have not mutated any OpenHands state yet, so a clean error is
    # safe to return.
    try:
        kc_user_id = await token_manager.create_keycloak_user(
            email=email,
            password=password,
            email_verified=True,
        )
    except KeycloakError as e:
        logger.warning(
            'provision_user:keycloak_create_failed',
            extra={
                'caller_user_id': caller_user_id,
                'target_org_id': str(target_org_id),
                'email': email,
                'error': str(e),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail='Failed to create Keycloak user (it may already exist)',
        ) from e
    except Exception:
        logger.exception(
            'provision_user:keycloak_create_unexpected',
            extra={
                'caller_user_id': caller_user_id,
                'target_org_id': str(target_org_id),
                'email': email,
            },
            stack_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to create Keycloak user',
        )

    # Anything after this point may need to roll back partial OpenHands
    # state (DB rows, LiteLLM team, OrgMember rows) **and** the Keycloak
    # user, to avoid leaving an OpenHands ``User``/personal-``Org``
    # pointing at a Keycloak ``sub`` that no longer exists. Progress is
    # tracked by the two local variables below so the unwind path
    # targets exactly the state that was created; see
    # ``_rollback_partial_provision`` for the unwind ordering.
    openhands_user_id: UUID | None = None
    target_membership_added = False
    try:
        # 2. Get an offline token for the new user and store it. This
        # mirrors what ``keycloak_offline_callback`` does at the end of
        # the interactive flow so the user is immediately usable for
        # backend operations.
        offline_token = await token_manager.request_offline_token(
            username=email, password=password
        )
        await token_manager.store_offline_token(
            user_id=kc_user_id, offline_token=offline_token
        )

        # 3. Create the OpenHands user. Mirrors ``keycloak_callback``'s
        # ``UserStore.create_user`` call but with email_verified
        # explicitly true. ``UserStore.create_user`` creates the
        # personal org, default settings, and the owner OrgMember row
        # for that personal org.
        user_info_dict = {
            'sub': kc_user_id,
            'email': email,
            'email_verified': True,
            'preferred_username': email,
        }
        new_user = await UserStore.create_user(kc_user_id, user_info_dict)
        if new_user is None:
            raise RuntimeError('UserStore.create_user returned None')
        # From here on, an unwind must remove the User row + personal
        # org cascade in addition to the Keycloak user.
        openhands_user_id = new_user.id

        # 4. Stamp TOS / verification / consent flags so the provisioned
        # user does not get bounced to the email-verification or TOS
        # interstitials on first login.
        await _set_user_provisioned_flags(kc_user_id)

        # 5. Add the user to the *target* org (separate from their
        # auto-created personal org). Skip this step when the
        # caller-selected org happens to be the user's brand-new
        # personal org — that membership was just created by
        # ``UserStore.create_user`` and re-adding it would violate the
        # ``(org_id, user_id)`` uniqueness constraint.
        if target_org_id != openhands_user_id:
            settings = await OrgService.create_litellm_integration(
                target_org_id, kc_user_id
            )
            llm_api_key_secret = settings.agent_settings.llm.api_key
            # ``api_key`` is typed ``str | SecretStr | None`` on the
            # SDK side; org_invitation_service.py handles it the same
            # way. Defaulting to empty string lets LiteLLM-disabled
            # deployments still create memberships.
            if llm_api_key_secret is None:
                llm_api_key = ''
            elif isinstance(llm_api_key_secret, SecretStr):
                llm_api_key = llm_api_key_secret.get_secret_value()
            else:
                llm_api_key = llm_api_key_secret

            role = await RoleStore.get_role_by_name(provisioned_role)
            if role is None:
                raise RuntimeError(f'Role {provisioned_role!r} not found in database')

            await OrgMemberStore.add_user_to_org(
                org_id=target_org_id,
                user_id=openhands_user_id,
                role_id=role.id,
                llm_api_key=llm_api_key,
                status='active',
                agent_settings_diff={},
                conversation_settings_diff={},
            )
            # Track separately from ``openhands_user_id`` so the unwind
            # can decide whether the personal-org cascade alone is
            # enough or whether the target-org membership has to be
            # removed first (see ``_rollback_partial_provision``).
            target_membership_added = True

        # 6. Mint an API key bound to the target org. We pass
        # ``org_id`` explicitly so the key is pinned to the org the
        # admin selected rather than the user's freshly created
        # personal org (``current_org_id`` defaults to the personal
        # org).
        api_key_store = ApiKeyStore.get_instance()
        api_key = await api_key_store.create_api_key(
            user_id=kc_user_id,
            name=api_key_name,
            org_id=target_org_id,
        )
    except HTTPException:
        # FastAPI HTTPException is intentional — surface as-is, but
        # still attempt to clean up whatever post-Keycloak state we
        # created so we do not orphan a half-created identity.
        await _rollback_partial_provision(
            token_manager=token_manager,
            kc_user_id=kc_user_id,
            openhands_user_id=openhands_user_id,
            target_org_id=target_org_id,
            target_membership_added=target_membership_added,
        )
        raise
    except Exception as e:
        logger.exception(
            'provision_user:post_keycloak_failure',
            extra={
                'caller_user_id': caller_user_id,
                'target_org_id': str(target_org_id),
                'kc_user_id': kc_user_id,
                'email': email,
            },
            stack_info=True,
        )
        await _rollback_partial_provision(
            token_manager=token_manager,
            kc_user_id=kc_user_id,
            openhands_user_id=openhands_user_id,
            target_org_id=target_org_id,
            target_membership_added=target_membership_added,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='Failed to finish provisioning user',
        ) from e

    logger.info(
        'provision_user:success',
        extra={
            'caller_user_id': caller_user_id,
            'provisioned_user_id': kc_user_id,
            'target_org_id': str(target_org_id),
            'provisioned_role': provisioned_role,
            # Intentionally omit email/password from the log line; the
            # full audit trail of who provisioned whom is captured by
            # caller_user_id + provisioned_user_id.
        },
    )

    return ProvisionUserResponse(
        email=email,
        password=password,
        api_key=api_key,
        user_id=kc_user_id,
        org_id=str(target_org_id),
        role=provisioned_role,
    )


async def _rollback_partial_provision(
    *,
    token_manager: TokenManager,
    kc_user_id: str,
    openhands_user_id: UUID | None,
    target_org_id: UUID,
    target_membership_added: bool,
) -> None:
    """Best-effort rollback of a partially completed provision.

    Runs on the unwind path of an already-failed request. Every step is
    wrapped individually because we never want a secondary cleanup
    failure to mask the original provisioning error — the underlying
    helpers each log their own diagnostics, so swallowing here is
    intentional.

    Unwind order matters:

    1. **Target-org membership.** If ``OrgMemberStore.add_user_to_org``
       succeeded, the user has memberships in *both* the personal org
       and the target org. ``OrgStore.delete_org_cascade`` only
       cascade-deletes the ``User`` row when the user is the sole
       orphan of the org being deleted — a surviving target-org
       membership would cause the cascade to just reassign
       ``current_org_id`` to the target org and leave the user row
       behind. So we drop the target-org membership first.

       The freshly minted API key (step 6) is bound to ``target_org_id``
       and there is no failure step *after* the API key insert: if
       ``create_api_key`` raises mid-INSERT the row is never committed,
       and if it returns successfully the route never reaches this
       unwind. So the unwind does not need an explicit
       ``ApiKeyStore.delete`` call.

    2. **Personal-org cascade.** ``delete_org_cascade(personal_org_id,
       requester_user_id=kc_user_id)`` wipes the personal ``Org`` row,
       the owner ``OrgMember`` row, the personal-org LiteLLM team,
       org-scoped tables (``api_keys WHERE org_id =
       personal_org_id``, ``conversation_metadata_saas``,
       ``billing_sessions``, etc.) and — because the user is now sole
       orphan after step 1 — the ``User`` row itself, in a single
       transaction. ``requester_user_id`` must equal the deleted
       user's ``id`` so the cascade treats this as a personal-org
       self-service deletion rather than raising
       ``OrphanedUserError``.

    3. **Keycloak user.** Last, so the local OpenHands identity is
       gone before we drop the upstream identity. Re-doing the
       Keycloak delete via the existing ``delete_keycloak_user``
       helper keeps that retry/logging behaviour consistent with
       interactive-flow cleanup elsewhere.
    """
    # 1. Target-org artifacts. Skip silently if the membership was
    # never inserted — there is nothing to undo.
    if openhands_user_id is not None and target_membership_added:
        try:
            await OrgMemberStore.remove_user_from_org(target_org_id, openhands_user_id)
        except Exception:
            logger.exception(
                'provision_user:rollback_remove_target_membership_failed',
                extra={
                    'kc_user_id': kc_user_id,
                    'target_org_id': str(target_org_id),
                    'openhands_user_id': str(openhands_user_id),
                },
                stack_info=True,
            )

    # 2. Personal-org cascade. Skip when ``UserStore.create_user`` had
    # not yet run — there are no DB rows to compensate, and
    # ``delete_org_cascade`` would just no-op against a missing org.
    if openhands_user_id is not None:
        try:
            await OrgStore.delete_org_cascade(
                openhands_user_id, requester_user_id=kc_user_id
            )
        except Exception:
            logger.exception(
                'provision_user:rollback_delete_personal_org_failed',
                extra={
                    'kc_user_id': kc_user_id,
                    'openhands_user_id': str(openhands_user_id),
                },
                stack_info=True,
            )

    # 3. Keycloak user. Always last; runs even when no OpenHands DB
    # state was created (e.g. failure between Keycloak create and
    # ``UserStore.create_user``).
    try:
        await token_manager.delete_keycloak_user(kc_user_id)
    except Exception:
        logger.debug(
            'provision_user:rollback_delete_keycloak_user_failed',
            extra={'kc_user_id': kc_user_id},
        )
