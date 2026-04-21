"""
Service for forwarding Git provider webhook events to the automation service.

This service is optimized for high-traffic scenarios:
1. Resolves Git org → OpenHands org_id (via cached OrgGitClaim lookup)
2. For personal repos, resolves to personal org (via cached provider→Keycloak mapping)
3. Forwards minimal payload to automation service (just org_id + payload)
4. Access control checks are deferred to automation execution time

Supports multiple Git providers (GitHub, GitLab, Bitbucket, etc.).

The lazy access control approach means:
- Most webhooks only do cached lookups + HTTP forward
- Membership checks only happen when an automation actually matches

Security notes:
- Uses AUTOMATION_WEBHOOK_SECRET (not provider webhook secret) for signing
- Negative results are cached to prevent DoS via repeated lookups
"""

import asyncio
import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import aiohttp
from integrations.resolver_org_router import resolve_org_for_repo
from server.auth.constants import (
    AUTOMATION_SERVICE_TIMEOUT,
    AUTOMATION_SERVICE_URL,
    AUTOMATION_WEBHOOK_SECRET,
)
from server.auth.token_manager import TokenManager

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.provider import ProviderType
from openhands.server.shared import sio

# Cache TTL constants
ORG_CLAIM_CACHE_TTL_SECONDS = 3600  # 1 hour for org claims (rarely change)
USER_ID_CACHE_TTL_SECONDS = 86400  # 24 hours for user ID mappings (never change)

# Cache key prefixes (provider is appended dynamically)
ORG_CLAIM_CACHE_PREFIX = 'automation:org_claim'
USER_ID_CACHE_PREFIX = 'automation:idp_to_kc_user'


@dataclass
class OrgContext:
    """Context for the resolved organization."""

    org_id: UUID
    git_org: str


class AutomationEventService:
    """
    Service for forwarding webhook events to the automation service.

    Optimized for high traffic with:
    - Redis caching for org claim lookups (1 hour TTL)
    - Redis caching for provider→Keycloak user ID mappings (24 hour TTL)
    - Lazy access control (membership checks deferred to execution time)

    Supports multiple Git providers (GitHub, GitLab, Bitbucket, etc.).
    """

    def __init__(self, token_manager: TokenManager):
        from server.auth.constants import AUTOMATION_EVENT_FORWARDING_ENABLED

        self.token_manager = token_manager

        # Fail fast if forwarding is enabled but misconfigured
        if AUTOMATION_EVENT_FORWARDING_ENABLED:
            if not AUTOMATION_SERVICE_URL:
                raise ValueError(
                    'AUTOMATION_EVENT_FORWARDING_ENABLED=true but '
                    'AUTOMATION_SERVICE_URL is not configured'
                )
            if not AUTOMATION_WEBHOOK_SECRET:
                raise ValueError(
                    'AUTOMATION_EVENT_FORWARDING_ENABLED=true but '
                    'AUTOMATION_WEBHOOK_SECRET is not configured'
                )

    async def forward_event(
        self,
        provider: ProviderType,
        payload: dict[str, Any],
        installation_id: int | str,
    ) -> None:
        """
        Forward a Git provider webhook event to the automation service.

        This is designed to be called as a fire-and-forget background task.
        The forward path is optimized for speed - only org resolution is done here.
        Access control checks are deferred to automation execution time.

        Args:
            provider: The Git provider type (e.g., GITHUB, GITLAB, BITBUCKET)
            payload: The raw webhook payload from the provider
            installation_id: The provider's installation/webhook ID
        """
        org_id: UUID | None = None
        try:
            # Resolve org context (org_id and git_org name) - uses Redis cache
            org_context = await self._resolve_org_context(provider, payload)
            if not org_context:
                return

            org_id = org_context.org_id

            # Build minimal payload and forward immediately
            # Access control is NOT computed here - it's deferred to execution time
            event_payload = self._build_event_payload(org_context, payload)
            await self._send_to_automation_service(provider, org_id, event_payload)

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # Network errors are expected and recoverable
            logger.error(
                f'[AutomationEventService] Network error forwarding '
                f'{provider.value} event (org_id={org_id}): {e}',
                exc_info=True,
                extra={'installation_id': installation_id},
            )
        except Exception as e:
            # Log unexpected errors. Note: This is a background task, so exceptions
            # won't surface to the HTTP caller - they're logged for debugging only.
            logger.error(
                f'[AutomationEventService] Unexpected error forwarding '
                f'{provider.value} event (org_id={org_id}): {e}',
                exc_info=True,
                extra={'installation_id': installation_id},
            )
            # Don't re-raise in background task - just log for debugging

    async def _resolve_org_context(
        self, provider: ProviderType, payload: dict[str, Any]
    ) -> OrgContext | None:
        """
        Resolve the organization context from the webhook payload.

        Uses Redis caching for both org claims and user ID mappings.
        Returns None if the org cannot be resolved (not claimed, no personal org).

        Args:
            provider: The Git provider type
            payload: The webhook payload from the provider
        """
        git_org_name, owner_type, owner_id = self._extract_owner_info(provider, payload)

        if not git_org_name:
            logger.warning(
                f'[AutomationEventService] No repository owner in '
                f'{provider.value} payload, skipping'
            )
            return None

        # Try to resolve via OrgGitClaim
        org_id = await self._resolve_git_org(provider, git_org_name)

        # Fallback for personal repos (owner_type indicates individual user)
        if not org_id and owner_type == 'User':
            org_id = await self._resolve_personal_org(provider, owner_id)
            if org_id:
                logger.info(
                    f'[AutomationEventService] Resolved personal repo owner '
                    f'{git_org_name} to personal org {org_id} ({provider.value})'
                )

        if not org_id:
            logger.warning(
                f'[AutomationEventService] {provider.value} org {git_org_name} '
                f'not claimed and no personal org found, skipping'
            )
            return None

        return OrgContext(org_id=org_id, git_org=git_org_name)

    def _extract_owner_info(
        self, provider: ProviderType, payload: dict[str, Any]
    ) -> tuple[str | None, str | None, int | None]:
        """
        Extract owner information from the webhook payload.

        Different providers structure their payloads differently, so this method
        normalizes the extraction.

        Args:
            provider: The Git provider type
            payload: The webhook payload

        Returns:
            Tuple of (git_org_name, owner_type, owner_id)
            - git_org_name: The organization/user name that owns the repo
            - owner_type: 'User' or 'Organization' (or provider-specific equivalent)
            - owner_id: The numeric ID of the owner (for personal org resolution)
        """
        # Compare using .value to handle different ProviderType enum instances
        # (e.g., test mocks may use a different enum class with the same values)
        if provider == ProviderType.GITHUB:
            repo = payload.get('repository', {})
            owner = repo.get('owner', {})
            return owner.get('login'), owner.get('type'), owner.get('id')

        logger.warning(f'Unsupported provider ({provider.value})')
        return None, None, None

    def _build_event_payload(
        self,
        org_context: OrgContext,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build the minimal event payload to forward to the automation service.

        Access control is NOT included here - it's deferred to execution time.
        This keeps the forward path fast for high-traffic scenarios.
        """
        return {
            'organization': {
                'git_org': org_context.git_org,
                'openhands_org_id': str(org_context.org_id),
            },
            'payload': payload,
        }

    # =========================================================================
    # Cached Org Resolution Methods
    # =========================================================================

    async def _resolve_git_org(
        self, provider: ProviderType, git_org_name: str
    ) -> UUID | None:
        """
        Resolve a Git organization name to an OpenHands org_id.

        Uses Redis caching with 1-hour TTL. Caches both positive and negative
        results to avoid repeated DB queries for unclaimed orgs.

        Args:
            provider: The Git provider type
            git_org_name: The organization/user name from the provider

        Note: Org names are normalized to lowercase for both cache keys and
        DB queries. This matches the OrgGitClaim schema which stores
        git_organization as lowercase.
        """
        normalized_org = git_org_name.lower()
        cache_key = f'{ORG_CLAIM_CACHE_PREFIX}:{provider.value}:{normalized_org}'

        # Check cache first
        cached = await self._get_cached_value(cache_key)
        if cached is not None:
            if cached == 'none':
                logger.debug(
                    f'[AutomationEventService] Cache hit (negative): '
                    f'{provider.value} org {git_org_name} not claimed'
                )
                return None
            logger.debug(
                f'[AutomationEventService] Cache hit: '
                f'{provider.value} org {git_org_name} -> {cached}'
            )
            return UUID(cached)

        # Cache miss - use resolve_org_for_repo without user_id (no membership check)
        # Construct a minimal repo name since resolve_org_for_repo extracts the org
        org_id = await resolve_org_for_repo(
            provider=provider.value,
            full_repo_name=f'{normalized_org}/',
        )

        # Cache the result (including negative results)
        if org_id:
            await self._set_cached_value(
                cache_key, str(org_id), ORG_CLAIM_CACHE_TTL_SECONDS
            )
            return org_id
        else:
            # Cache negative result to avoid repeated DB queries
            await self._set_cached_value(cache_key, 'none', ORG_CLAIM_CACHE_TTL_SECONDS)
            return None

    async def _resolve_personal_org(
        self, provider: ProviderType, provider_user_id: int | str | None
    ) -> UUID | None:
        """
        Resolve a provider user to their personal OpenHands org.

        For personal repos (owner type is 'User'), the OpenHands org_id
        is the user's keycloak user ID. This allows users to set up
        automations on their personal repos without needing an OrgGitClaim.

        Uses Redis caching for the provider→Keycloak user ID mapping (24h TTL).

        Args:
            provider: The Git provider type
            provider_user_id: The user ID from the provider (numeric or string UUID)
        """
        if not provider_user_id:
            return None

        keycloak_id = await self._get_keycloak_user_id_cached(
            provider, provider_user_id
        )
        if keycloak_id:
            return UUID(keycloak_id)
        return None

    async def _get_keycloak_user_id_cached(
        self, provider: ProviderType, provider_user_id: int | str
    ) -> str | None:
        """
        Convert a provider user ID to a Keycloak user ID.

        Uses Redis caching with 24-hour TTL since this mapping never changes.
        Caches negative results to avoid repeated Keycloak queries.

        Args:
            provider: The Git provider type
            provider_user_id: The user ID from the provider
        """
        cache_key = f'{USER_ID_CACHE_PREFIX}:{provider.value}:{provider_user_id}'

        # Check cache first
        cached = await self._get_cached_value(cache_key)
        if cached is not None:
            if cached == 'none':
                logger.debug(
                    f'[AutomationEventService] Cache hit (negative): '
                    f'{provider.value} user {provider_user_id} not in Keycloak'
                )
                return None
            logger.debug(
                f'[AutomationEventService] Cache hit: '
                f'{provider.value} user {provider_user_id} -> Keycloak {cached}'
            )
            return cached

        # Cache miss - query Keycloak
        try:
            keycloak_id = await self.token_manager.get_user_id_from_idp_user_id(
                str(provider_user_id), provider
            )

            # Cache the result (including negative results)
            if keycloak_id:
                await self._set_cached_value(
                    cache_key, keycloak_id, USER_ID_CACHE_TTL_SECONDS
                )
            else:
                # Cache negative result to prevent repeated Keycloak queries
                await self._set_cached_value(
                    cache_key, 'none', USER_ID_CACHE_TTL_SECONDS
                )

            return keycloak_id
        except Exception as e:
            # Log at warning level to surface programmer errors and API issues
            logger.warning(
                f'[AutomationEventService] Failed to get keycloak ID for '
                f'{provider.value} user {provider_user_id}: {e}'
            )
            return None

    # =========================================================================
    # Generic Redis Cache Helpers
    # =========================================================================

    async def _get_cached_value(self, cache_key: str) -> str | None:
        """
        Get a cached value from Redis.

        Returns the cached string value, or None if not cached or Redis unavailable.
        Falls back to DB/API queries if Redis is unavailable (graceful degradation).

        Warning: When Redis is unavailable, every webhook will hit the DB directly.
        Monitor logs for 'Redis unavailable' warnings to detect degradation.
        """
        try:
            redis = getattr(sio.manager, 'redis', None)
            if not redis:
                # Log at warning level - this is a significant degradation that
                # will cause DB load. Monitor these logs for alerting.
                logger.warning(
                    '[AutomationEventService] Redis unavailable for cache read, '
                    'falling back to direct DB queries (this will increase DB load)'
                )
                return None

            cached = await redis.get(cache_key)
            if cached is None:
                return None

            # Redis returns bytes, decode to string
            return cached.decode('utf-8') if isinstance(cached, bytes) else cached
        except Exception as e:
            # Log at warning level - cache errors cause DB fallback
            logger.warning(
                f'[AutomationEventService] Redis cache read error '
                f'(falling back to DB): {e}'
            )
            return None

    async def _set_cached_value(
        self, cache_key: str, value: str, ttl_seconds: int
    ) -> None:
        """
        Set a cached value in Redis with TTL.

        Fails silently if Redis is unavailable (graceful degradation).
        """
        try:
            redis = getattr(sio.manager, 'redis', None)
            if not redis:
                # Silent failure - read path already logs the warning
                return

            await redis.setex(cache_key, ttl_seconds, value)
        except Exception as e:
            # Log at warning level for visibility
            logger.warning(f'[AutomationEventService] Redis cache write error: {e}')

    def _sign_payload(self, payload_bytes: bytes) -> str:
        """
        Sign a payload using the dedicated automation shared secret.

        Uses AUTOMATION_WEBHOOK_SECRET (not GitHub webhook secret) to maintain
        separate trust boundaries between GitHub webhooks and internal services.

        Returns the signature in the format 'sha256=<hex_digest>'.
        """
        signature = hmac.new(
            AUTOMATION_WEBHOOK_SECRET.encode('utf-8'),
            msg=payload_bytes,
            digestmod=hashlib.sha256,
        ).hexdigest()
        return f'sha256={signature}'

    async def _send_to_automation_service(
        self,
        provider: ProviderType,
        org_id: UUID,
        payload: dict[str, Any],
    ) -> None:
        """
        Send the normalized payload to the automation service.

        The payload is signed using AUTOMATION_WEBHOOK_SECRET so the
        automation service can verify it came from the OpenHands server.

        Args:
            provider: The Git provider type
            org_id: The OpenHands organization ID
            payload: The event payload to send
        """
        if not AUTOMATION_SERVICE_URL:
            logger.warning(
                '[AutomationEventService] AUTOMATION_SERVICE_URL not configured'
            )
            return

        # Build endpoint URL. AUTOMATION_SERVICE_URL may include path segments
        # (e.g., https://example.com/api/automation), so we strip trailing slash
        # and append our path. The provider is included in the URL path.
        base_url = AUTOMATION_SERVICE_URL.rstrip('/')
        url = f'{base_url}/v1/events/{org_id}/{provider.value}'

        # Serialize payload to JSON bytes for signing
        payload_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
        signature = self._sign_payload(payload_bytes)

        headers = {
            'Content-Type': 'application/json',
            'X-Hub-Signature-256': signature,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    data=payload_bytes,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=AUTOMATION_SERVICE_TIMEOUT),
                ) as resp:
                    if resp.status >= 400:
                        # Try JSON first (expected interface), fall back to text
                        # for infrastructure errors (502/503 from load balancer)
                        try:
                            body = await resp.json()
                        except (aiohttp.ContentTypeError, ValueError):
                            body = await resp.text()
                        logger.warning(
                            f'[AutomationEventService] Automation service returned '
                            f'{resp.status} for {provider.value} org {org_id}: {body}'
                        )
                    else:
                        data = await resp.json()
                        matched = data.get('matched', 0)
                        logger.info(
                            f'[AutomationEventService] Forwarded {provider.value} '
                            f'event to org {org_id}: {matched} automations matched'
                        )
        except asyncio.TimeoutError:
            logger.warning(
                f'[AutomationEventService] Timeout ({AUTOMATION_SERVICE_TIMEOUT}s) '
                f'forwarding {provider.value} event to automation service'
            )
        except aiohttp.ClientError as e:
            logger.warning(
                f'[AutomationEventService] HTTP error forwarding '
                f'{provider.value} event to automation service: {e}'
            )
