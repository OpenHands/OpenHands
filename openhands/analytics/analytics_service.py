"""Core analytics service for OpenHands.

Provides a thin wrapper around the PostHog SDK with:
- Consent gate: all calls are no-ops when consented=False
- OSS/SaaS dual-mode: $process_person_profile is set to False in OSS mode;
  set_person_properties and group_identify are SaaS-only
- Common properties: app_mode, is_feature_env added to every event
- Feature-env distinct_id prefix: FEATURE_ prefix for staging/feature envs
- SDK error isolation: all exceptions are caught and logged, never raised

This module must NOT import from enterprise/. It receives all configuration
via constructor args.
"""

from datetime import datetime, timezone
from typing import Any

from posthog import Posthog

from openhands.analytics.analytics_constants import (
    CONVERSATION_CREATED,
    CONVERSATION_DELETED,
    CONVERSATION_ERRORED,
    CONVERSATION_FINISHED,
    CREDIT_LIMIT_REACHED,
    CREDIT_PURCHASED,
    GIT_PROVIDER_CONNECTED,
    ONBOARDING_COMPLETED,
    SETTINGS_SAVED,
    TEAM_MEMBERS_INVITED,
    TRAJECTORY_DOWNLOADED,
    USER_LOGGED_IN,
    USER_SIGNED_UP,
)
from openhands.analytics.analytics_context import AnalyticsContext
from openhands.app_server.utils.logger import openhands_logger as logger
from openhands.server.types import AppMode


class AnalyticsService:
    """Server-side analytics service backed by PostHog.

    Args:
        api_key: PostHog project API key. Pass an empty string to disable.
        host: PostHog ingest host URL.
        app_mode: AppMode.OPENHANDS (OSS) or AppMode.SAAS.
        is_feature_env: True when running in a feature/staging environment.
    """

    def __init__(
        self,
        api_key: str,
        host: str,
        app_mode: AppMode,
        is_feature_env: bool,
    ) -> None:
        self._app_mode = app_mode
        self._is_feature_env = is_feature_env
        self._client: Posthog = Posthog(
            project_api_key=api_key,
            host=host,
            disabled=not api_key,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def capture(
        self,
        ctx: AnalyticsContext,
        event: str,
        properties: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> None:
        """Capture a server-side event.

        Consent gate: returns immediately when ctx.consented=False.
        Common properties (app_mode, is_feature_env, and optionally org_id /
        $session_id / $process_person_profile) are merged with caller-provided
        properties before forwarding to PostHog.
        """
        if not ctx.consented:
            return

        merged = self._common_properties(org_id=ctx.org_id, session_id=session_id)
        if properties:
            merged.update(properties)

        try:
            self._client.capture(
                distinct_id=self._distinct_id(ctx.user_id),
                event=event,
                properties=merged,
            )
        except Exception:
            logger.exception('AnalyticsService.capture failed for event=%s', event)

    def set_person_properties(
        self,
        ctx: AnalyticsContext,
        properties: dict[str, Any],
    ) -> None:
        """Set person properties in PostHog (SaaS-only).

        No-op in OSS mode or when ctx.consented=False.
        """
        if not ctx.consented:
            return
        if self._app_mode != AppMode.SAAS:
            return

        try:
            self._client.set(
                distinct_id=self._distinct_id(ctx.user_id),
                properties=properties,
            )
        except Exception:
            logger.exception('AnalyticsService.set_person_properties failed')

    def group_identify(
        self,
        ctx: AnalyticsContext,
        group_type: str,
        group_key: str,
        properties: dict[str, Any],
    ) -> None:
        """Associate a group with properties (SaaS-only).

        No-op in OSS mode or when ctx.consented=False.
        """
        if not ctx.consented:
            return
        if self._app_mode != AppMode.SAAS:
            return

        try:
            self._client.group_identify(
                group_type=group_type,
                group_key=group_key,
                properties=properties,
                distinct_id=self._distinct_id(ctx.user_id),
            )
        except Exception:
            logger.exception('AnalyticsService.group_identify failed')

    # ------------------------------------------------------------------
    # Typed event methods
    # ------------------------------------------------------------------

    def track_user_signed_up(
        self,
        ctx: AnalyticsContext,
        *,
        email_domain: str | None = None,
        invitation_source: str = 'self_signup',
        session_id: str | None = None,
    ) -> None:
        """Track 'user signed up' event.

        Fired when a new user completes registration.
        """
        self.capture(
            ctx=ctx,
            event=USER_SIGNED_UP,
            properties={
                'email_domain': email_domain,
                'invitation_source': invitation_source,
            },
            session_id=session_id,
        )

    def track_user_logged_in(
        self,
        ctx: AnalyticsContext,
        *,
        idp: str,
        session_id: str | None = None,
    ) -> None:
        """Track 'user logged in' event.

        Fired when an existing user authenticates.
        """
        self.capture(
            ctx=ctx,
            event=USER_LOGGED_IN,
            properties={
                'idp': idp,
            },
            session_id=session_id,
        )

    def track_conversation_created(
        self,
        ctx: AnalyticsContext,
        *,
        conversation_id: str,
        trigger: str | None = None,
        llm_model: str | None = None,
        agent_type: str = 'default',
        has_repository: bool = False,
        session_id: str | None = None,
    ) -> None:
        """Track 'conversation created' event.

        Fired when a new conversation is started.
        """
        self.capture(
            ctx=ctx,
            event=CONVERSATION_CREATED,
            properties={
                'conversation_id': conversation_id,
                'trigger': trigger,
                'llm_model': llm_model,
                'agent_type': agent_type,
                'has_repository': has_repository,
            },
            session_id=session_id,
        )

    def track_conversation_finished(
        self,
        ctx: AnalyticsContext,
        *,
        conversation_id: str,
        terminal_state: str,
        turn_count: int | None = None,
        accumulated_cost_usd: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        llm_model: str | None = None,
        trigger: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Track 'conversation finished' event.

        Fired when a conversation reaches a terminal state.
        """
        self.capture(
            ctx=ctx,
            event=CONVERSATION_FINISHED,
            properties={
                'conversation_id': conversation_id,
                'terminal_state': terminal_state,
                'turn_count': turn_count,
                'accumulated_cost_usd': accumulated_cost_usd,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'llm_model': llm_model,
                'trigger': trigger,
            },
            session_id=session_id,
        )

    def track_conversation_errored(
        self,
        ctx: AnalyticsContext,
        *,
        conversation_id: str,
        error_type: str,
        error_message: str | None = None,
        llm_model: str | None = None,
        turn_count: int | None = None,
        terminal_state: str,
        session_id: str | None = None,
    ) -> None:
        """Track 'conversation errored' event.

        Fired when a conversation ends in an error state.
        """
        self.capture(
            ctx=ctx,
            event=CONVERSATION_ERRORED,
            properties={
                'conversation_id': conversation_id,
                'error_type': error_type,
                'error_message': error_message,
                'llm_model': llm_model,
                'turn_count': turn_count,
                'terminal_state': terminal_state,
            },
            session_id=session_id,
        )

    def track_conversation_deleted(
        self,
        ctx: AnalyticsContext,
        *,
        conversation_id: str,
        session_id: str | None = None,
    ) -> None:
        """Track 'conversation deleted' event.

        Fired when a user deletes a conversation.
        """
        self.capture(
            ctx=ctx,
            event=CONVERSATION_DELETED,
            properties={
                'conversation_id': conversation_id,
            },
            session_id=session_id,
        )

    def track_credit_purchased(
        self,
        ctx: AnalyticsContext,
        *,
        amount_usd: float,
        credit_balance_before: float | None = None,
        credit_balance_after: float | None = None,
        session_id: str | None = None,
    ) -> None:
        """Track 'credit purchased' event.

        Fired when a user completes a credit purchase.
        """
        self.capture(
            ctx=ctx,
            event=CREDIT_PURCHASED,
            properties={
                'amount_usd': amount_usd,
                'credit_balance_before': credit_balance_before,
                'credit_balance_after': credit_balance_after,
            },
            session_id=session_id,
        )

    def track_credit_limit_reached(
        self,
        ctx: AnalyticsContext,
        *,
        conversation_id: str,
        credit_balance: float | None = None,
        llm_model: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Track 'credit limit reached' event.

        Fired when a conversation is blocked by insufficient credits.
        """
        self.capture(
            ctx=ctx,
            event=CREDIT_LIMIT_REACHED,
            properties={
                'conversation_id': conversation_id,
                'credit_balance': credit_balance,
                'llm_model': llm_model,
            },
            session_id=session_id,
        )

    def track_git_provider_connected(
        self,
        ctx: AnalyticsContext,
        *,
        provider_type: str,
        session_id: str | None = None,
    ) -> None:
        """Track 'git provider connected' event.

        Fired when a user connects a git provider (GitHub, GitLab, etc.).
        """
        self.capture(
            ctx=ctx,
            event=GIT_PROVIDER_CONNECTED,
            properties={
                'provider_type': provider_type,
            },
            session_id=session_id,
        )

    def track_onboarding_completed(
        self,
        ctx: AnalyticsContext,
        *,
        selections: dict[str, str | list[str]] | None = None,
        session_id: str | None = None,
    ) -> None:
        """Track 'onboarding completed' event.

        Fired when a user finishes the onboarding flow.

        Args:
            selections: Dynamic key-value pairs from the onboarding form.
                Keys are question IDs (e.g., 'role', 'org_size', 'use_case',
                'org_name', 'org_domain'). Values are the selected option IDs
                or arrays for multi-select questions.
        """
        self.capture(
            ctx=ctx,
            event=ONBOARDING_COMPLETED,
            properties=selections or {},
            session_id=session_id,
        )

    def track_settings_saved(
        self,
        ctx: AnalyticsContext,
        *,
        settings_changed: list[str] | None = None,
        session_id: str | None = None,
    ) -> None:
        """Track 'settings saved' event.

        Fired when a user saves their settings.
        """
        self.capture(
            ctx=ctx,
            event=SETTINGS_SAVED,
            properties={
                'settings_changed': settings_changed,
            },
            session_id=session_id,
        )

    def track_trajectory_downloaded(
        self,
        ctx: AnalyticsContext,
        *,
        conversation_id: str,
        session_id: str | None = None,
    ) -> None:
        """Track 'trajectory downloaded' event.

        Fired when a user downloads a conversation trajectory.
        """
        self.capture(
            ctx=ctx,
            event=TRAJECTORY_DOWNLOADED,
            properties={
                'conversation_id': conversation_id,
            },
            session_id=session_id,
        )

    def track_team_members_invited(
        self,
        ctx: AnalyticsContext,
        *,
        invited_count: int,
        successful_count: int,
        failed_count: int,
        role: str,
        session_id: str | None = None,
    ) -> None:
        """Track 'team members invited' event.

        Fired when a user invites team members to their organization.
        """
        self.capture(
            ctx=ctx,
            event=TEAM_MEMBERS_INVITED,
            properties={
                'invited_count': invited_count,
                'successful_count': successful_count,
                'failed_count': failed_count,
                'role': role,
            },
            session_id=session_id,
        )

    def identify_user(
        self,
        ctx: AnalyticsContext,
        *,
        email: str | None = None,
        org_name: str | None = None,
        idp: str | None = None,
        orgs: list[dict[str, Any]] | None = None,
    ) -> None:
        """Identify a user and their org memberships in PostHog.

        Consolidates the duplicated ``set_person_properties`` +
        ``group_identify`` pattern from auth.py and oauth_device.py into
        a single call.

        Consent gate: returns immediately when ``ctx.consented=False``.
        SaaS gate: returns immediately in OSS mode (person profiles are
        SaaS-only).

        Args:
            ctx: Analytics context with user_id, org_id, and consent.
            email: User email address.
            org_name: Current org display name.
            idp: Identity provider (e.g. ``"github"``, ``"google"``).
            orgs: List of org dicts with keys ``id``, ``name``,
                  ``member_count`` for group_identify calls.
        """
        if not ctx.consented:
            return
        if self._app_mode != AppMode.SAAS:
            return

        try:
            # Person properties
            self.set_person_properties(
                ctx=ctx,
                properties={
                    'email': email,
                    'org_id': ctx.org_id,
                    'org_name': org_name,
                    'plan_tier': None,
                    'idp': idp,
                    'last_login_at': datetime.now(timezone.utc).isoformat(),
                },
            )

            # Group identify for each org membership
            if orgs:
                for org in orgs:
                    self.group_identify(
                        ctx=ctx,
                        group_type='org',
                        group_key=org['id'],
                        properties={
                            'org_name': org.get('name'),
                            'plan_tier': None,
                            'created_at': None,
                            'member_count': org.get('member_count'),
                        },
                    )
        except Exception:
            logger.exception('AnalyticsService.identify_user failed')

    def shutdown(self) -> None:
        """Flush and shut down the PostHog client.

        Safe to call multiple times. SDK errors are logged, not raised.
        """
        try:
            self._client.shutdown()
        except Exception:
            logger.exception('AnalyticsService.shutdown failed')

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _distinct_id(self, user_id: str) -> str:
        """Return the PostHog distinct_id for the given user.

        In feature/staging environments, prefixes with 'FEATURE_' to keep
        test traffic separate from production profiles.
        """
        if self._is_feature_env:
            return f'FEATURE_{user_id}'
        return user_id

    def _common_properties(
        self,
        org_id: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Build the base property dict included on every event."""
        props: dict[str, Any] = {
            'app_mode': self._app_mode.value,
            'is_feature_env': self._is_feature_env,
        }

        if org_id is not None:
            props['org_id'] = org_id

        if session_id is not None:
            props['$session_id'] = session_id

        # PostHog person profiles are not useful in OSS mode (no user accounts)
        if self._app_mode != AppMode.SAAS:
            props['$process_person_profile'] = False

        return props
