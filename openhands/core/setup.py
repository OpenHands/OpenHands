# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import hashlib
import os
import uuid

from pydantic import SecretStr

from openhands.core.config import (
    OpenHandsConfig,
)
from openhands.integrations.provider import (
    ProviderToken,
    ProviderType,
)
from openhands.storage.data_models.secrets import Secrets


def get_provider_tokens():
    """Retrieve provider tokens from environment variables and return them as a dictionary.

    Returns:
        A dictionary mapping ProviderType to ProviderToken if tokens are found, otherwise None.
    """
    # Collect provider tokens from environment variables if available
    provider_tokens = {}
    if 'GITHUB_TOKEN' in os.environ:
        github_token = SecretStr(os.environ['GITHUB_TOKEN'])
        provider_tokens[ProviderType.GITHUB] = ProviderToken(token=github_token)

    if 'GITLAB_TOKEN' in os.environ:
        gitlab_token = SecretStr(os.environ['GITLAB_TOKEN'])
        provider_tokens[ProviderType.GITLAB] = ProviderToken(token=gitlab_token)

    if 'BITBUCKET_TOKEN' in os.environ:
        bitbucket_token = SecretStr(os.environ['BITBUCKET_TOKEN'])
        provider_tokens[ProviderType.BITBUCKET] = ProviderToken(token=bitbucket_token)

    # Forgejo support (e.g., Codeberg or self-hosted Forgejo)
    if 'FORGEJO_TOKEN' in os.environ:
        forgejo_token = SecretStr(os.environ['FORGEJO_TOKEN'])
        # If a base URL is provided, extract the domain to use as host override
        forgejo_base_url = os.environ.get('FORGEJO_BASE_URL', '').strip()
        host: str | None = None
        if forgejo_base_url:
            # Normalize by stripping protocol and any path (e.g., /api/v1)
            url = forgejo_base_url
            if url.startswith(('http://', 'https://')):
                try:
                    from urllib.parse import urlparse

                    parsed = urlparse(url)
                    host = parsed.netloc or None
                except Exception:
                    pass
            if host is None:
                host = url.replace('https://', '').replace('http://', '')
            host = host.split('/')[0].strip('/') if host else None
        provider_tokens[ProviderType.FORGEJO] = ProviderToken(
            token=forgejo_token, host=host
        )

    # Wrap provider tokens in Secrets if any tokens were found
    secret_store = (
        Secrets(provider_tokens=provider_tokens) if provider_tokens else None  # type: ignore[arg-type]
    )
    return secret_store.provider_tokens if secret_store else None


def generate_sid(config: OpenHandsConfig, session_name: str | None = None) -> str:
    """Generate a session id based on the session name and the jwt secret.

    The session ID is kept short to ensure Kubernetes resource names don't exceed
    the 63-character limit when prefixed with 'openhands-runtime-' (18 chars).
    Total length is limited to 32 characters to allow for suffixes like '-svc', '-pvc'.
    """
    session_name = session_name or str(uuid.uuid4())
    jwt_secret = config.jwt_secret

    hash_str = hashlib.sha256(f'{session_name}{jwt_secret}'.encode('utf-8')).hexdigest()

    # Limit total session ID length to 32 characters for Kubernetes compatibility:
    # - 'openhands-runtime-' (18 chars) + session_id (32 chars) = 50 chars
    # - Leaves 13 chars for suffixes like '-svc' (4), '-pvc' (4), '-ingress-code' (13)
    if len(session_name) > 16:
        # If session_name is too long, use first 16 chars + 15-char hash for better readability
        # e.g., "vscode-extension" -> "vscode-extensio-{15-char-hash}"
        session_id = f'{session_name[:16]}-{hash_str[:15]}'
    else:
        # If session_name is short enough, use it + remaining space for hash
        remaining_chars = 32 - len(session_name) - 1  # -1 for the dash
        session_id = f'{session_name}-{hash_str[:remaining_chars]}'

    return session_id[:32]  # Ensure we never exceed 32 characters
