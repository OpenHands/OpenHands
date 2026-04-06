# TODO(OpenHands/evaluation#418): Delete this file and import directly from
# openhands.sdk.utils.redact once openhands-sdk >1.16.1 is released.
# These functions are copied from the SDK's redact.py to unblock PRs while
# waiting for the next SDK release.
#
# Source of truth: openhands-sdk/openhands/sdk/utils/redact.py
#   in repo: https://github.com/OpenHands/software-agent-sdk

import copy
import re
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from openhands.sdk.utils.redact import sanitize_dict

# ---------------------------------------------------------------------------
# URL param redaction
# ---------------------------------------------------------------------------

SENSITIVE_URL_PARAMS = frozenset(
    {
        'tavilyapikey',
        'apikey',
        'api_key',
        'token',
        'access_token',
        'secret',
        'key',
    }
)


def _is_secret_key(key: str) -> bool:
    key_upper = key.upper()
    return any(
        p in key_upper
        for p in (
            'AUTHORIZATION',
            'COOKIE',
            'CREDENTIAL',
            'KEY',
            'PASSWORD',
            'SECRET',
            'SESSION',
            'TOKEN',
        )
    )


def redact_url_params(url: str) -> str:
    """Redact sensitive query parameter values from a URL string."""
    try:
        parsed = urlparse(url)
    except Exception:
        return url
    if not parsed.query:
        return url
    params = parse_qs(parsed.query, keep_blank_values=True)
    redacted_params: dict[str, list[str]] = {}
    for param_name, values in params.items():
        if param_name.lower() in SENSITIVE_URL_PARAMS or _is_secret_key(param_name):
            redacted_params[param_name] = ['<redacted>'] * len(values)
        else:
            redacted_params[param_name] = values
    redacted_query = urlencode(redacted_params, doseq=True)
    return urlunparse(parsed._replace(query=redacted_query))


def _walk_redact_urls(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _walk_redact_urls(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_redact_urls(item) for item in obj]
    if isinstance(obj, str) and '?' in obj:
        return redact_url_params(obj)
    return obj


# ---------------------------------------------------------------------------
# sanitize_config
# ---------------------------------------------------------------------------


def sanitize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy a config dict, redact secret keys and URL query params."""
    config = copy.deepcopy(config)
    config = sanitize_dict(config)
    config = _walk_redact_urls(config)
    return config


# ---------------------------------------------------------------------------
# Text / string redaction
# ---------------------------------------------------------------------------

_API_KEY_LITERAL_RE = re.compile(
    r'\b('
    # OpenRouter / OpenAI / Anthropic
    r'sk-(?:or-v1|proj|ant-(?:api|oat)\d{2})-[A-Za-z0-9_-]{20,}'
    r'|gsk_[A-Za-z0-9]{20,}'  # GROQ
    r'|hf_[A-Za-z0-9]{20,}'  # HuggingFace
    r'|tgp_v1_[A-Za-z0-9_-]{20,}'  # Together AI
    r'|ghp_[A-Za-z0-9]{20,}'  # GitHub PAT (classic)
    r'|github_pat_[A-Za-z0-9_]{20,}'  # GitHub PAT (fine-grained)
    r'|sk-oh-[A-Za-z0-9]{20,}'  # OpenHands session tokens
    r'|ctx7sk-[A-Za-z0-9_-]{10,}'  # Context7 MCP keys
    r'|cla_[A-Za-z0-9_-]{20,}'  # Claude.ai MCP tokens
    r'|sntryu_[A-Za-z0-9]{10,}'  # Sentry tokens
    r'|lin_api_[A-Za-z0-9]{10,}'  # Linear API tokens
    r'|tvly-[A-Za-z0-9_-]{10,}'  # Tavily keys
    r'|ATATT3x[A-Za-z0-9_-]{10,}'  # Jira/Atlassian tokens
    r'|xoxb-[A-Za-z0-9_-]{20,}'  # Slack bot tokens
    r'|xoxp-[A-Za-z0-9_-]{20,}'  # Slack user tokens
    r'|Bearer\s+[A-Za-z0-9_.-]{20,}'  # Bearer tokens
    r')'
)


def redact_api_key_literals(text: str) -> str:
    """Replace bare API key literals from common providers with <redacted>."""
    return _API_KEY_LITERAL_RE.sub('<redacted>', text)


def redact_text_secrets(text: str) -> str:
    """Redact secrets from a string representation of a config object."""
    # api_key='...' patterns
    text = re.sub(r"api_key='[^']*'", "api_key='<redacted>'", text)
    text = re.sub(r'api_key="[^"]*"', 'api_key="<redacted>"', text)

    # Dict entries with sensitive key names
    text = re.sub(
        r"('[A-Z_]*(?:KEY|SECRET|TOKEN|PASSWORD)[A-Z_]*':\s*')[^']*(')",
        r'\g<1><redacted>\2',
        text,
    )
    text = re.sub(
        r'("[A-Z_]*(?:KEY|SECRET|TOKEN|PASSWORD)[A-Z_]*":\s*")[^"]*(")',
        r'\g<1><redacted>\2',
        text,
    )

    # URL query params
    text = re.sub(
        r'((?:tavilyApiKey|apiKey|api_key|token|access_token|secret|key)=)'
        r"[^&\s'\")\]]+",
        r'\g<1><redacted>',
        text,
        flags=re.IGNORECASE,
    )

    # Authorization header values
    text = re.sub(
        r"('Authorization':\s*')[^']*(')",
        r'\g<1><redacted>\2',
        text,
    )

    # X-Session-API-Key header values
    text = re.sub(
        r"('X-Session-API-Key':\s*')[^']*(')",
        r'\g<1><redacted>\2',
        text,
    )

    # Bare API key literals
    text = redact_api_key_literals(text)

    return text
