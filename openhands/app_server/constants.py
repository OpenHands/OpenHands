"""Constants for the OpenHands App Server.

This module contains constants that are used across the app server,
including security-related configurations for secret name validation.
"""

import os
from collections.abc import Mapping

# =============================================================================
# SECRET LIMITS (configurable via environment variables)
# =============================================================================

# Maximum number of secrets that can be passed via API in a single request.
# Prevents abuse by limiting the size of the secrets dictionary.
# Override with: OH_MAX_API_SECRETS_COUNT
MAX_API_SECRETS_COUNT: int = int(os.getenv('OH_MAX_API_SECRETS_COUNT', '50'))

# Maximum length of a secret name in characters.
# Environment variable names should be concise; this prevents excessively long names.
# Override with: OH_MAX_API_SECRET_NAME_LENGTH
MAX_API_SECRET_NAME_LENGTH: int = int(os.getenv('OH_MAX_API_SECRET_NAME_LENGTH', '256'))

# Maximum length of a secret value in bytes.
# 64KB is generous for API keys/tokens while preventing massive payloads.
# Override with: OH_MAX_API_SECRET_VALUE_LENGTH
MAX_API_SECRET_VALUE_LENGTH: int = int(
    os.getenv('OH_MAX_API_SECRET_VALUE_LENGTH', '65536')
)


# =============================================================================
# SECRET NAME VALIDATION
# =============================================================================

# -----------------------------------------------------------------------------
# BLOCKED: These names CANNOT be used as user-provided secrets.
#
# These environment variables are injected into the agent-server container
# at startup. User-provided secrets with these names would override them
# when exported in bash commands, potentially breaking the sandbox or
# creating security vulnerabilities.
# -----------------------------------------------------------------------------
BLOCKED_SECRET_NAMES: frozenset[str] = frozenset(
    {
        # Agent-server container configuration (from initial_env)
        'OPENVSCODE_SERVER_ROOT',
        'OH_ENABLE_VNC',
        'LOG_JSON',
        'OH_CONVERSATIONS_PATH',
        'OH_BASH_EVENTS_DIR',
        'PYTHONUNBUFFERED',
        'ENV_LOG_LEVEL',
        # Webhook and CORS - overriding could redirect callbacks to malicious endpoints
        'OH_WEBHOOKS_0_BASE_URL',
        'OH_ALLOW_CORS_ORIGINS_0',
        # Worker ports - could break web application functionality
        'WORKER_1',
        'WORKER_2',
    }
)

# -----------------------------------------------------------------------------
# BLOCKED PREFIXES: Secret names starting with these prefixes are blocked.
#
# LLM_* variables are auto-forwarded to the agent-server container to enforce
# LLM controls (timeouts, retries, model restrictions, etc.). Allowing users
# to override these would let them escape app-server LLM controls.
# -----------------------------------------------------------------------------
BLOCKED_SECRET_PREFIXES: tuple[str, ...] = ('LLM_',)

# -----------------------------------------------------------------------------
# OVERRIDABLE: These are system-provided but users MAY override them.
# Documented here for clarity - these are explicitly ALLOWED, not blocked.
#
# Use case: User wants to use their own credentials instead of the
# organization-level credentials provided by the system.
# -----------------------------------------------------------------------------
OVERRIDABLE_SYSTEM_SECRETS: frozenset[str] = frozenset(
    {
        # Git Provider Tokens - users may provide their own credentials
        # Note: Provider tokens are fetched via app-server API, not container env
        'GITHUB_TOKEN',
        'GITLAB_TOKEN',
        'BITBUCKET_TOKEN',
        'AZURE_DEVOPS_TOKEN',
        'FORGEJO_TOKEN',
        # AWS Credentials - used for Bedrock LLM access
        # Users may want to use their own AWS account for Bedrock models
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'AWS_REGION_NAME',
    }
)


def validate_secret_name(name: str) -> None:
    """Validate that a secret name is allowed.

    Args:
        name: The secret name to validate

    Raises:
        ValueError: If the name is blocked (exact match or prefix match),
                    or exceeds the maximum length
    """
    # Check name length
    if len(name) > MAX_API_SECRET_NAME_LENGTH:
        raise ValueError(
            f'Secret name exceeds maximum length of {MAX_API_SECRET_NAME_LENGTH} characters '
            f'(got {len(name)}). Configure via OH_MAX_API_SECRET_NAME_LENGTH.'
        )

    upper_name = name.upper()

    # Check exact matches
    if upper_name in BLOCKED_SECRET_NAMES:
        raise ValueError(
            f"Secret name '{name}' is reserved for internal use and cannot be overridden. "
            f'See openhands.app_server.constants for the list of blocked names.'
        )

    # Check prefix matches
    for prefix in BLOCKED_SECRET_PREFIXES:
        if upper_name.startswith(prefix):
            raise ValueError(
                f"Secret name '{name}' starts with reserved prefix '{prefix}' and cannot be used. "
                f'These variables are used for LLM configuration controls.'
            )

    # Note: OVERRIDABLE_SYSTEM_SECRETS are intentionally allowed


def validate_secrets_dict(secrets: Mapping[str, object] | None) -> None:
    """Validate the entire secrets dictionary for size limits.

    This should be called before iterating over individual secrets.

    Args:
        secrets: The secrets dictionary to validate (can be None).
                 Values can be str or SecretStr (uses get_secret_value()).

    Raises:
        ValueError: If the dictionary exceeds size limits
    """
    if secrets is None:
        return

    # Check number of secrets
    if len(secrets) > MAX_API_SECRETS_COUNT:
        raise ValueError(
            f'Too many secrets provided: {len(secrets)} exceeds maximum of '
            f'{MAX_API_SECRETS_COUNT}. Configure via OH_MAX_API_SECRETS_COUNT.'
        )

    # Check individual value lengths
    for name, value in secrets.items():
        # Handle both str and SecretStr (Pydantic's SecretStr has get_secret_value())
        if hasattr(value, 'get_secret_value'):
            value_str = value.get_secret_value()  # type: ignore[union-attr]
        else:
            value_str = str(value)
        value_bytes = len(value_str.encode('utf-8'))
        if value_bytes > MAX_API_SECRET_VALUE_LENGTH:
            raise ValueError(
                f"Secret '{name}' value exceeds maximum length of "
                f'{MAX_API_SECRET_VALUE_LENGTH} bytes (got {value_bytes}). '
                f'Configure via OH_MAX_API_SECRET_VALUE_LENGTH.'
            )
