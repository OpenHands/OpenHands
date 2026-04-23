"""Tests for Settings._rewrite_volatile_ollama_url_in_settings model validator.

These tests verify that ephemeral 172.16.0.0/12 Docker-bridge IPs stored as
llm_base_url are transparently rewritten to host.docker.internal when the
Settings model is loaded/instantiated.
"""

import pytest

from openhands.storage.data_models.settings import Settings


def _s(model: str, base_url: str | None) -> Settings:
    return Settings(llm_model=model, llm_base_url=base_url)


@pytest.mark.parametrize(
    'model, input_url, expected_url',
    [
        # prefixed ollama model + volatile IP → rewritten
        (
            'ollama/qwen2.5-coder:14b',
            'http://172.28.174.246:11434',
            'http://host.docker.internal:11434',
        ),
        # bare model with tag (no http prefix) → treated as Ollama → rewritten
        (
            'qwen2.5-coder:14b',
            'http://172.17.0.1:11434',
            'http://host.docker.internal:11434',
        ),
        # volatile IP + path → path must be preserved
        (
            'ollama/mistral',
            'http://172.20.0.1:11434/v1',
            'http://host.docker.internal:11434/v1',
        ),
        # non-volatile IP → unchanged
        (
            'ollama/llama3',
            'http://192.168.1.50:11434',
            'http://192.168.1.50:11434',
        ),
        # already host.docker.internal → unchanged
        (
            'ollama/mistral',
            'http://host.docker.internal:11434',
            'http://host.docker.internal:11434',
        ),
        # volatile IP but wrong port (not 11434) → unchanged
        (
            'ollama/mistral',
            'http://172.20.0.1:8080',
            'http://172.20.0.1:8080',
        ),
        # non-Ollama model with volatile-looking IP → unchanged
        (
            'gpt-4o',
            'http://172.28.0.1:11434',
            'http://172.28.0.1:11434',
        ),
    ],
)
def test_settings_volatile_url_rewrite(
    model: str, input_url: str, expected_url: str
) -> None:
    s = _s(model, input_url)
    assert s.llm_base_url == expected_url


def test_settings_no_base_url_remains_none() -> None:
    s = _s('ollama/llama3', None)
    assert s.llm_base_url is None
