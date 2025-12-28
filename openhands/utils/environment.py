from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

LEMONADE_DOCKER_BASE_URL = 'http://host.docker.internal:8000/api/v1/'
_LEMONADE_PROVIDER_NAME = 'lemonade'
_LEMONADE_MODEL_PREFIX = 'lemonade/'


@lru_cache(maxsize=1)
def is_running_in_docker() -> bool:
    """Best-effort detection for Docker containers."""
    docker_env_markers = (
        Path('/.dockerenv'),
        Path('/run/.containerenv'),
    )
    if any(marker.exists() for marker in docker_env_markers):
        return True

    if os.environ.get('DOCKER_CONTAINER') == 'true':
        return True

    try:
        with Path('/proc/self/cgroup').open('r', encoding='utf-8') as cgroup_file:
            for line in cgroup_file:
                if any(token in line for token in ('docker', 'containerd', 'kubepods')):
                    return True
    except FileNotFoundError:
        pass

    return False


def is_lemonade_provider(
    model: str | None,
    custom_provider: str | None = None,
) -> bool:
    provider = (custom_provider or '').strip().lower()
    if provider == _LEMONADE_PROVIDER_NAME:
        return True
    return (model or '').startswith(_LEMONADE_MODEL_PREFIX)


def get_effective_llm_base_url(
    model: str | None,
    base_url: str | None,
    custom_provider: str | None = None,
) -> str | None:
    """Return the runtime LLM base URL with provider-specific overrides."""
    if (
        base_url in (None, '')
        and is_lemonade_provider(model, custom_provider)
        and is_running_in_docker()
    ):
        return LEMONADE_DOCKER_BASE_URL
    return base_url
