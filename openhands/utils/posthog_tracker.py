"""PostHog tracking utilities for OpenHands events."""

import os

from openhands.core.logger import openhands_logger as logger

# Lazy import posthog to avoid import errors in environments where it's not installed
posthog = None


def _init_posthog():
    """Initialize PostHog client lazily."""
    global posthog
    if posthog is None:
        try:
            import posthog as ph

            posthog = ph
            posthog.api_key = os.environ.get(
                'POSTHOG_CLIENT_KEY', 'phc_3ESMmY9SgqEAGBB6sMGK5ayYHkeUuknH2vP6FmWH9RA'
            )
            posthog.host = os.environ.get('POSTHOG_HOST', 'https://us.i.posthog.com')
        except ImportError:
            logger.warning(
                'PostHog not installed. Analytics tracking will be disabled.'
            )
            posthog = None


def track_agent_task_completed(
    conversation_id: str,
    user_id: str | None = None,
    app_mode: str | None = None,
) -> None:
    """Track when an agent completes a task.

    Args:
        conversation_id: The ID of the conversation/session
        user_id: The ID of the user (optional, may be None for unauthenticated users)
        app_mode: The application mode (saas/oss), optional
    """
    _init_posthog()

    if posthog is None:
        return

    # Use conversation_id as distinct_id if user_id is not available
    # This ensures we can track completions even for anonymous users
    distinct_id = user_id if user_id else f'conversation_{conversation_id}'

    try:
        posthog.capture(
            distinct_id=distinct_id,
            event='agent_task_completed',
            properties={
                'conversation_id': conversation_id,
                'user_id': user_id,
                'app_mode': app_mode or 'unknown',
            },
        )
        logger.debug(
            'posthog_track',
            extra={
                'event': 'agent_task_completed',
                'conversation_id': conversation_id,
                'user_id': user_id,
            },
        )
    except Exception as e:
        logger.warning(
            f'Failed to track agent_task_completed to PostHog: {e}',
            extra={
                'conversation_id': conversation_id,
                'error': str(e),
            },
        )


def track_user_signup_completed(
    user_id: str,
    signup_timestamp: str,
) -> None:
    """Track when a user completes signup by accepting TOS.

    Args:
        user_id: The ID of the user (Keycloak user ID)
        signup_timestamp: ISO format timestamp of when TOS was accepted
    """
    _init_posthog()

    if posthog is None:
        return

    try:
        posthog.capture(
            distinct_id=user_id,
            event='user_signup_completed',
            properties={
                'user_id': user_id,
                'signup_timestamp': signup_timestamp,
            },
        )
        logger.debug(
            'posthog_track',
            extra={
                'event': 'user_signup_completed',
                'user_id': user_id,
            },
        )
    except Exception as e:
        logger.warning(
            f'Failed to track user_signup_completed to PostHog: {e}',
            extra={
                'user_id': user_id,
                'error': str(e),
            },
        )


def track_credit_limit_reached(
    conversation_id: str,
    user_id: str | None = None,
    current_budget: float = 0.0,
    max_budget: float = 0.0,
) -> None:
    """Track when a user reaches their credit limit during a conversation.

    Args:
        conversation_id: The ID of the conversation/session
        user_id: The ID of the user (optional, may be None for unauthenticated users)
        current_budget: The current budget spent
        max_budget: The maximum budget allowed
    """
    _init_posthog()

    if posthog is None:
        return

    distinct_id = user_id if user_id else f'conversation_{conversation_id}'

    try:
        posthog.capture(
            distinct_id=distinct_id,
            event='credit_limit_reached',
            properties={
                'conversation_id': conversation_id,
                'user_id': user_id,
                'current_budget': current_budget,
                'max_budget': max_budget,
            },
        )
        logger.debug(
            'posthog_track',
            extra={
                'event': 'credit_limit_reached',
                'conversation_id': conversation_id,
                'user_id': user_id,
                'current_budget': current_budget,
                'max_budget': max_budget,
            },
        )
    except Exception as e:
        logger.warning(
            f'Failed to track credit_limit_reached to PostHog: {e}',
            extra={
                'conversation_id': conversation_id,
                'error': str(e),
            },
        )


def track_credits_purchased(
    user_id: str,
    amount_usd: float,
    credits_added: float,
    stripe_session_id: str,
) -> None:
    """Track when a user successfully purchases credits.

    Args:
        user_id: The ID of the user (Keycloak user ID)
        amount_usd: The amount paid in USD (cents converted to dollars)
        credits_added: The number of credits added to the user's account
        stripe_session_id: The Stripe checkout session ID
    """
    _init_posthog()

    if posthog is None:
        return

    try:
        posthog.capture(
            distinct_id=user_id,
            event='credits_purchased',
            properties={
                'user_id': user_id,
                'amount_usd': amount_usd,
                'credits_added': credits_added,
                'stripe_session_id': stripe_session_id,
            },
        )
        logger.debug(
            'posthog_track',
            extra={
                'event': 'credits_purchased',
                'user_id': user_id,
                'amount_usd': amount_usd,
                'credits_added': credits_added,
            },
        )
    except Exception as e:
        logger.warning(
            f'Failed to track credits_purchased to PostHog: {e}',
            extra={
                'user_id': user_id,
                'error': str(e),
            },
        )


def alias_user_identities(
    keycloak_user_id: str,
    git_login: str,
) -> None:
    """Alias a user's Keycloak ID with their git provider login for unified tracking.

    This allows PostHog to link events tracked from the frontend (using git provider login)
    with events tracked from the backend (using Keycloak user ID).

    PostHog Python alias syntax: alias(previous_id, distinct_id)
    - previous_id: The old/previous distinct ID that will be merged
    - distinct_id: The new/canonical distinct ID to merge into

    For our use case:
    - Git provider login is the previous_id (first used in frontend, before backend auth)
    - Keycloak user ID is the distinct_id (canonical backend ID)
    - Result: All events with git login will be merged into Keycloak user ID

    Args:
        keycloak_user_id: The Keycloak user ID (canonical distinct_id)
        git_login: The git provider username (GitHub/GitLab/Bitbucket) to merge

    Reference:
        https://github.com/PostHog/posthog-python/blob/master/posthog/client.py
    """
    _init_posthog()

    if posthog is None:
        return

    try:
        # Merge git provider login into Keycloak user ID
        # posthog.alias(previous_id, distinct_id) - official Python SDK signature
        posthog.alias(git_login, keycloak_user_id)
        logger.debug(
            'posthog_alias',
            extra={
                'previous_id': git_login,
                'distinct_id': keycloak_user_id,
            },
        )
    except Exception as e:
        logger.warning(
            f'Failed to alias user identities in PostHog: {e}',
            extra={
                'keycloak_user_id': keycloak_user_id,
                'git_login': git_login,
                'error': str(e),
            },
        )
