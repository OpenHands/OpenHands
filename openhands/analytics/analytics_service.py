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
    CONVERSATION_RESUMED,
    CREDIT_LIMIT_REACHED,
    CREDIT_PURCHASED,
    ENTERPRISE_LEAD_FORM_SUBMITTED,
    GIT_PROVIDER_CONNECTED,
    MCP_CONFIG_UPDATED,
    ONBOARDING_COMPLETED,
    SAAS_SELFHOSTED_INQUIRY,
    SETTINGS_SAVED,
    TEAM_MEMBERS_INVITED,
    TRAJECTORY_DOWNLOADED,
    USER_ACTIVATED,
    USER_LOGGED_IN,
    USER_SIGNED_UP,
)
from openhands.core.logger import openhands_logger as logger
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
        distinct_id: str,
        event: str,
        properties: dict[str, Any] | None = None,
        org_id: str | None = None,
        session_id: str | None = None,
        consented: bool = True,
    ) -> None:
        """Capture a server-side event.

        Consent gate: returns immediately when consented=False.
        Common properties (app_mode, is_feature_env, and optionally org_id /
        $session_id / $process_person_profile) are merged with caller-provided
        properties before forwarding to PostHog.
        """
        if not consented:
            return

        merged = self._common_properties(org_id=org_id, session_id=session_id)
        if properties:
            merged.update(properties)

        try:
            self._client.capture(
                distinct_id=self._distinct_id(distinct_id),
                event=event,
                properties=merged,
            )
        except Exception:
            logger.exception('AnalyticsService.capture failed for event=%s', event)

    def set_person_properties(
        self,
        distinct_id: str,
        properties: dict[str, Any],
        consented: bool = True,
    ) -> None:
        """Set person properties in PostHog (SaaS-only).

        No-op in OSS mode or when consented=False.
        """
        if not consented:
            return
        if self._app_mode != AppMode.SAAS:
            return

        try:
            self._client.set(
                distinct_id=self._distinct_id(distinct_id),
                properties=properties,
            )
        except Exception:
            logger.exception('AnalyticsService.set_person_properties failed')

    def group_identify(
        self,
        group_type: str,
        group_key: str,
        properties: dict[str, Any],
        distinct_id: str | None = None,
        consented: bool = True,
    ) -> None:
        """Associate a group with properties (SaaS-only).

        No-op in OSS mode or when consented=False.
        """
        if not consented:
            return
        if self._app_mode != AppMode.SAAS:
            return

        try:
            kwargs: dict[str, Any] = {
                'group_type': group_type,
                'group_key': group_key,
                'properties': properties,
            }
            if distinct_id is not None:
                kwargs['distinct_id'] = self._distinct_id(distinct_id)
            self._client.group_identify(**kwargs)
        except Exception:
            logger.exception('AnalyticsService.group_identify failed')

    # ------------------------------------------------------------------
    # Typed event methods
    # ------------------------------------------------------------------

    def track_user_signed_up(
        self,
        distinct_id: str,
        *,
        idp: str,
        email_domain: str | None = None,
        invitation_source: str = 'self_signup',
        org_id: str | None = None,
        session_id: str | None = None,
        consented: bool = True,
    ) -> None:
        """Track 'user signed up' event.

        Fired when a new user completes registration.
        """
        self.capture(
            distinct_id=distinct_id,
            event=USER_SIGNED_UP,
            properties={
                'idp': idp,
                'email_domain': email_domain,
                'invitation_source': invitation_source,
            },
            org_id=org_id,
            session_id=session_id,
            consented=consented,
        )

    def track_user_logged_in(
        self,
        distinct_id: str,
        *,
        idp: str,
        org_id: str | None = None,
        session_id: str | None = None,
        consented: bool = True,
    ) -> None:
        """Track 'user logged in' event.

        Fired when an existing user authenticates.
        """
        self.capture(
            distinct_id=distinct_id,
            event=USER_LOGGED_IN,
            properties={
                'idp': idp,
            },
            org_id=org_id,
            session_id=session_id,
            consented=consented,
        )

    def track_conversation_created(
        self,
        distinct_id: str,
        *,
        conversation_id: str,
        trigger: str | None = None,
        llm_model: str | None = None,
        agent_type: str = 'default',
        has_repository: bool = False,
        org_id: str | None = None,
        session_id: str | None = None,
        consented: bool = True,
    ) -> None:
        """Track 'conversation created' event.

        Fired when a new conversation is started.
        """
        self.capture(
            distinct_id=distinct_id,
            event=CONVERSATION_CREATED,
            properties={
                'conversation_id': conversation_id,
                'trigger': trigger,
                'llm_model': llm_model,
                'agent_type': agent_type,
                'has_repository': has_repository,
            },
            org_id=org_id,
            session_id=session_id,
            consented=consented,
        )

    def track_conversation_finished(
        self,
        distinct_id: str,
        *,
        conversation_id: str,
        terminal_state: str,
        turn_count: int | None = None,
        accumulated_cost_usd: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        llm_model: str | None = None,
        trigger: str | None = None,
        org_id: str | None = None,
        session_id: str | None = None,
        consented: bool = True,
    ) -> None:
        """Track 'conversation finished' event.

        Fired when a conversation reaches a terminal state.
        """
        self.capture(
            distinct_id=distinct_id,
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
            org_id=org_id,
            session_id=session_id,
            consented=consented,
        )

    def track_conversation_errored(
        self,
        distinct_id: str,
        *,
        conversation_id: str,
        error_type: str,
        error_message: str | None = None,
        llm_model: str | None = None,
        turn_count: int | None = None,
        terminal_state: str,
        org_id: str | None = None,
        session_id: str | None = None,
        consented: bool = True,
    ) -> None:
        """Track 'conversation errored' event.

        Fired when a conversation ends in an error state.
        """
        self.capture(
            distinct_id=distinct_id,
            event=CONVERSATION_ERRORED,
            properties={
                'conversation_id': conversation_id,
                'error_type': error_type,
                'error_message': error_message,
                'llm_model': llm_model,
                'turn_count': turn_count,
                'terminal_state': terminal_state,
            },
            org_id=org_id,
            session_id=session_id,
            consented=consented,
        )

    def track_conversation_deleted(
        self,
        distinct_id: str,
        *,
        conversation_id: str,
        org_id: str | None = None,
        session_id: str | None = None,
        consented: bool = True,
    ) -> None:
        """Track 'conversation deleted' event.

        Fired when a user deletes a conversation.
        """
        self.capture(
            distinct_id=distinct_id,
            event=CONVERSATION_DELETED,
            properties={
                'conversation_id': conversation_id,
            },
            org_id=org_id,
            session_id=session_id,
            consented=consented,
        )

    def track_conversation_resumed(
        self,
        distinct_id: str,
        *,
        sandbox_id: str,
        org_id: str | None = None,
        session_id: str | None = None,
        consented: bool = True,
    ) -> None:
        """Track 'conversation resumed' event.

        Fired when a user resumes a paused sandbox.
        """
        self.capture(
            distinct_id=distinct_id,
            event=CONVERSATION_RESUMED,
            properties={
                'sandbox_id': sandbox_id,
            },
            org_id=org_id,
            session_id=session_id,
            consented=consented,
        )

    def track_credit_purchased(
        self,
        distinct_id: str,
        *,
        amount_usd: float,
        credit_balance_before: float | None = None,
        credit_balance_after: float | None = None,
        org_id: str | None = None,
        session_id: str | None = None,
        consented: bool = True,
    ) -> None:
        """Track 'credit purchased' event.

        Fired when a user completes a credit purchase.
        """
        self.capture(
            distinct_id=distinct_id,
            event=CREDIT_PURCHASED,
            properties={
                'amount_usd': amount_usd,
                'credit_balance_before': credit_balance_before,
                'credit_balance_after': credit_balance_after,
            },
            org_id=org_id,
            session_id=session_id,
            consented=consented,
        )

    def track_credit_limit_reached(
        self,
        distinct_id: str,
        *,
        conversation_id: str,
        credit_balance: float | None = None,
        llm_model: str | None = None,
        org_id: str | None = None,
        session_id: str | None = None,
        consented: bool = True,
    ) -> None:
        """Track 'credit limit reached' event.

        Fired when a conversation is blocked by insufficient credits.
        """
        self.capture(
            distinct_id=distinct_id,
            event=CREDIT_LIMIT_REACHED,
            properties={
                'conversation_id': conversation_id,
                'credit_balance': credit_balance,
                'llm_model': llm_model,
            },
            org_id=org_id,
            session_id=session_id,
            consented=consented,
        )

    def track_user_activated(
        self,
        distinct_id: str,
        *,
        conversation_id: str,
        time_to_activate_seconds: float | None = None,
        llm_model: str | None = None,
        trigger: str | None = None,
        org_id: str | None = None,
        session_id: str | None = None,
        consented: bool = True,
    ) -> None:
        """Track 'user activated' event.

        Fired when a user completes their first successful conversation.
        """
        self.capture(
            distinct_id=distinct_id,
            event=USER_ACTIVATED,
            properties={
                'conversation_id': conversation_id,
                'time_to_activate_seconds': time_to_activate_seconds,
                'llm_model': llm_model,
                'trigger': trigger,
            },
            org_id=org_id,
            session_id=session_id,
            consented=consented,
        )

    def track_git_provider_connected(
        self,
        distinct_id: str,
        *,
        provider_type: str,
        org_id: str | None = None,
        session_id: str | None = None,
        consented: bool = True,
    ) -> None:
        """Track 'git provider connected' event.

        Fired when a user connects a git provider (GitHub, GitLab, etc.).
        """
        self.capture(
            distinct_id=distinct_id,
            event=GIT_PROVIDER_CONNECTED,
            properties={
                'provider_type': provider_type,
            },
            org_id=org_id,
            session_id=session_id,
            consented=consented,
        )

    def track_onboarding_completed(
        self,
        distinct_id: str,
        *,
        selections: dict[str, str | list[str]] | None = None,
        org_id: str | None = None,
        session_id: str | None = None,
        consented: bool = True,
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
            distinct_id=distinct_id,
            event=ONBOARDING_COMPLETED,
            properties=selections or {},
            org_id=org_id,
            session_id=session_id,
            consented=consented,
        )

    def track_saas_selfhosted_inquiry(
        self,
        distinct_id: str,
        *,
        location: str,
        org_id: str | None = None,
        session_id: str | None = None,
        consented: bool = True,
    ) -> None:
        """Track 'saas selfhosted inquiry' event.

        Fired when a user clicks 'Learn More' on an enterprise CTA.
        """
        self.capture(
            distinct_id=distinct_id,
            event=SAAS_SELFHOSTED_INQUIRY,
            properties={
                'location': location,
            },
            org_id=org_id,
            session_id=session_id,
            consented=consented,
        )

    def track_enterprise_lead_form_submitted(
        self,
        distinct_id: str,
        *,
        request_type: str,
        name: str,
        company: str,
        email: str,
        message: str,
        org_id: str | None = None,
        session_id: str | None = None,
        consented: bool = True,
    ) -> None:
        """Track 'enterprise lead form submitted' event.

        Fired when a user submits the enterprise contact form.
        """
        self.capture(
            distinct_id=distinct_id,
            event=ENTERPRISE_LEAD_FORM_SUBMITTED,
            properties={
                'request_type': request_type,
                'name': name,
                'company': company,
                'email': email,
                'message': message,
            },
            org_id=org_id,
            session_id=session_id,
            consented=consented,
        )

    def track_settings_saved(
        self,
        distinct_id: str,
        *,
        settings_changed: list[str] | None = None,
        org_id: str | None = None,
        session_id: str | None = None,
        consented: bool = True,
    ) -> None:
        """Track 'settings saved' event.

        Fired when a user saves their settings.
        """
        self.capture(
            distinct_id=distinct_id,
            event=SETTINGS_SAVED,
            properties={
                'settings_changed': settings_changed,
            },
            org_id=org_id,
            session_id=session_id,
            consented=consented,
        )

    def track_mcp_config_updated(
        self,
        distinct_id: str,
        *,
        has_mcp_config: bool = True,
        sse_servers_count: int = 0,
        stdio_servers_count: int = 0,
        org_id: str | None = None,
        session_id: str | None = None,
        consented: bool = True,
    ) -> None:
        """Track 'mcp config updated' event.

        Fired when a user updates their MCP server configuration.
        """
        self.capture(
            distinct_id=distinct_id,
            event=MCP_CONFIG_UPDATED,
            properties={
                'has_mcp_config': has_mcp_config,
                'sse_servers_count': sse_servers_count,
                'stdio_servers_count': stdio_servers_count,
            },
            org_id=org_id,
            session_id=session_id,
            consented=consented,
        )

    def track_trajectory_downloaded(
        self,
        distinct_id: str,
        *,
        conversation_id: str,
        org_id: str | None = None,
        session_id: str | None = None,
        consented: bool = True,
    ) -> None:
        """Track 'trajectory downloaded' event.

        Fired when a user downloads a conversation trajectory.
        """
        self.capture(
            distinct_id=distinct_id,
            event=TRAJECTORY_DOWNLOADED,
            properties={
                'conversation_id': conversation_id,
            },
            org_id=org_id,
            session_id=session_id,
            consented=consented,
        )

    def track_team_members_invited(
        self,
        distinct_id: str,
        *,
        org_id: str,
        invited_count: int,
        successful_count: int,
        failed_count: int,
        role: str,
        session_id: str | None = None,
        consented: bool = True,
    ) -> None:
        """Track 'team members invited' event.

        Fired when a user invites team members to their organization.
        """
        self.capture(
            distinct_id=distinct_id,
            event=TEAM_MEMBERS_INVITED,
            properties={
                'invited_count': invited_count,
                'successful_count': successful_count,
                'failed_count': failed_count,
                'role': role,
            },
            org_id=org_id,
            session_id=session_id,
            consented=consented,
        )

    def identify_user(
        self,
        distinct_id: str,
        consented: bool = True,
        email: str | None = None,
        org_id: str | None = None,
        org_name: str | None = None,
        idp: str | None = None,
        orgs: list[dict[str, Any]] | None = None,
    ) -> None:
        """Identify a user and their org memberships in PostHog.

        Consolidates the duplicated ``set_person_properties`` +
        ``group_identify`` pattern from auth.py and oauth_device.py into
        a single call.

        Consent gate: returns immediately when ``consented=False``.
        SaaS gate: returns immediately in OSS mode (person profiles are
        SaaS-only).

        Args:
            distinct_id: User ID string.
            consented: Whether user has opted in to analytics.
            email: User email address.
            org_id: Current org ID string.
            org_name: Current org display name.
            idp: Identity provider (e.g. ``"github"``, ``"google"``).
            orgs: List of org dicts with keys ``id``, ``name``,
                  ``member_count`` for group_identify calls.
        """
        if not consented:
            return
        if self._app_mode != AppMode.SAAS:
            return

        try:
            # Person properties
            self.set_person_properties(
                distinct_id=distinct_id,
                properties={
                    'email': email,
                    'org_id': org_id,
                    'org_name': org_name,
                    'plan_tier': None,
                    'idp': idp,
                    'last_login_at': datetime.now(timezone.utc).isoformat(),
                },
                consented=consented,
            )

            # Group identify for each org membership
            if orgs:
                for org in orgs:
                    self.group_identify(
                        group_type='org',
                        group_key=org['id'],
                        properties={
                            'org_name': org.get('name'),
                            'plan_tier': None,
                            'created_at': None,
                            'member_count': org.get('member_count'),
                        },
                        distinct_id=distinct_id,
                        consented=consented,
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
