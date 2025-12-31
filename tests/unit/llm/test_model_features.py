import pytest

from openhands.llm.model_features import (
    ModelFeatures,
    get_features,
    model_matches,
    normalize_model_name,
)


@pytest.mark.parametrize(
    'raw,expected',
    [
        ('  OPENAI/gpt-4o  ', 'gpt-4o'),
        ('anthropic/claude-3-7-sonnet', 'claude-3-7-sonnet'),
        ('litellm_proxy/gemini-2.5-pro', 'gemini-2.5-pro'),
        ('qwen3-coder-480b-a35b-instruct', 'qwen3-coder-480b-a35b-instruct'),
        ('gpt-5', 'gpt-5'),
        ('deepseek/DeepSeek-R1-0528:671b-Q4_K_XL', 'deepseek-r1-0528'),
        ('openai/GLM-4.5-GGUF', 'glm-4.5'),
        ('openrouter/gpt-4o-mini', 'gpt-4o-mini'),
        (
            'bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0',
            'anthropic.claude-3-5-sonnet-20241022-v2',
        ),
        ('', ''),
        (None, ''),  # type: ignore[arg-type]
    ],
)
def test_normalize_model_name(raw, expected):
    assert normalize_model_name(raw) == expected


@pytest.mark.parametrize(
    'name,pattern,expected',
    [
        ('gpt-4o', 'gpt-4o*', True),
        ('openai/gpt-4o', 'gpt-4o*', True),
        ('litellm_proxy/gpt-4o-mini', 'gpt-4o*', True),
        ('claude-3-7-sonnet-20250219', 'claude-3-7-sonnet*', True),
        ('o1-2024-12-17', 'o1*', True),
        ('grok-4-0709', 'grok-4-0709', True),
        ('grok-4-0801', 'grok-4-0709', False),
    ],
)
def test_model_matches(name, pattern, expected):
    assert model_matches(name, [pattern]) is expected


@pytest.mark.parametrize(
    'name,pattern,expected',
    [
        ('openai/gpt-4o', 'openai/gpt-4o*', True),
        ('openrouter/gpt-4o', 'openai/gpt-4o*', False),
        ('litellm_proxy/gpt-4o-mini', 'litellm_proxy/gpt-4o*', True),
        (
            'gpt-4o',
            'openai/gpt-4o*',
            False,
        ),  # basename alone should not match provider-qualified
        ('unknown-model', 'gpt-5*', False),
    ],
)
def test_model_matches_provider_qualified(name, pattern, expected):
    assert model_matches(name, [pattern]) is expected


@pytest.mark.parametrize(
    'model,expect',
    [
        (
            'gpt-4o',
            ModelFeatures(
                supports_function_calling=True,
                supports_reasoning_effort=False,
                supports_prompt_cache=False,
                supports_stop_words=True,
            ),
        ),
        (
            'gpt-5',
            ModelFeatures(
                supports_function_calling=True,
                supports_reasoning_effort=True,
                supports_prompt_cache=False,
                supports_stop_words=True,
            ),
        ),
        (
            'gpt-5-mini-2025-08-07',
            ModelFeatures(
                supports_function_calling=True,
                supports_reasoning_effort=True,
                supports_prompt_cache=False,
                supports_stop_words=True,
            ),
        ),
        (
            'o3-mini',
            ModelFeatures(
                supports_function_calling=True,
                supports_reasoning_effort=True,
                supports_prompt_cache=False,
                supports_stop_words=True,
            ),
        ),
        (
            'o1-2024-12-17',
            ModelFeatures(
                supports_function_calling=True,
                supports_reasoning_effort=True,
                supports_prompt_cache=False,
                supports_stop_words=False,
            ),
        ),
        (
            'xai/grok-4-0709',
            ModelFeatures(
                supports_function_calling=False,
                supports_reasoning_effort=False,
                supports_prompt_cache=False,
                supports_stop_words=False,
            ),
        ),
        (
            'anthropic/claude-3-7-sonnet',
            ModelFeatures(
                supports_function_calling=True,
                supports_reasoning_effort=False,
                supports_prompt_cache=True,
                supports_stop_words=True,
            ),
        ),
        (
            'litellm_proxy/claude-3.7-sonnet',
            ModelFeatures(
                supports_function_calling=True,
                supports_reasoning_effort=False,
                supports_prompt_cache=True,
                supports_stop_words=True,
            ),
        ),
        (
            'gemini-2.5-pro',
            ModelFeatures(
                supports_function_calling=True,
                supports_reasoning_effort=True,
                supports_prompt_cache=False,
                supports_stop_words=True,
            ),
        ),
        (
            'openai/gpt-4o',
            ModelFeatures(
                supports_function_calling=True,
                supports_reasoning_effort=False,
                supports_prompt_cache=False,
                supports_stop_words=True,
            ),
        ),  # provider-qualified still matches basename patterns
    ],
)
def test_get_features(model, expect):
    features = get_features(model)
    assert features == expect


@pytest.mark.parametrize(
    'model',
    [
        # Anthropic families
        'claude-3-7-sonnet-20250219',
        'claude-3.7-sonnet',
        'claude-sonnet-3-7-latest',
        'claude-3-5-sonnet',
        'claude-3.5-haiku',
        'claude-3-5-haiku-20241022',
        'claude-sonnet-4-latest',
        'claude-opus-4-1-20250805',
        # OpenAI families
        'gpt-4o',
        'gpt-4.1',
        'gpt-5',
        'gpt-5-mini-2025-08-07',
        # o-series
        'o1-2024-12-17',
        'o3-mini',
        'o4-mini',
        # Google Gemini
        'gemini-2.5-pro',
        'gemini-3-pro-preview',
        # Others
        'kimi-k2-0711-preview',
        'kimi-k2-instruct',
        'qwen3-coder',
        'qwen3-coder-480b-a35b-instruct',
    ],
)
def test_function_calling_models(model):
    features = get_features(model)
    assert features.supports_function_calling is True


@pytest.mark.parametrize(
    'model',
    [
        'o1-2024-12-17',
        'o3-mini',
        'o4-mini',
        'gemini-2.5-flash',
        'gemini-2.5-pro',
        'gpt-5',
        'gpt-5-mini-2025-08-07',
    ],
)
def test_reasoning_effort_models(model):
    features = get_features(model)
    assert features.supports_reasoning_effort is True


@pytest.mark.parametrize(
    'model',
    [
        'deepseek/DeepSeek-R1-0528:671b-Q4_K_XL',
        'DeepSeek-R1-0528',
    ],
)
def test_deepseek_reasoning_effort_models(model):
    features = get_features(model)
    assert features.supports_reasoning_effort is True


@pytest.mark.parametrize(
    'model',
    [
        'claude-3-7-sonnet-20250219',
        'claude-3.7-sonnet',
        'claude-sonnet-3-7-latest',
        'claude-3-5-sonnet',
        'claude-3-5-haiku-20241022',
        'claude-3-haiku-20240307',
        'claude-3-opus-20240229',
        'claude-sonnet-4-latest',
    ],
)
def test_prompt_cache_models(model):
    features = get_features(model)
    assert features.supports_prompt_cache is True


@pytest.mark.parametrize(
    'model,expected',
    [
        # Positive cases: exactly those supported on main
        ('o1', True),
        ('o1-2024-12-17', True),
        ('o3', True),
        ('o3-2025-04-16', True),
        ('o3-mini', True),
        ('o3-mini-2025-01-31', True),
        ('o4-mini', True),
        ('o4-mini-2025-04-16', True),
        ('gemini-2.5-flash', True),
        ('gemini-2.5-pro', True),
        ('gpt-5', True),
        ('gpt-5-2025-08-07', True),
        ('gpt-5-mini-2025-08-07', True),
        ('claude-opus-4-1-20250805', False),
        # DeepSeek
        ('deepseek/DeepSeek-R1-0528:671b-Q4_K_XL', True),
        ('DeepSeek-R1-0528', True),
        # Negative cases: ensure we didn't unintentionally expand
        ('o1-mini', False),
        ('o1-preview', False),
        ('gemini-1.0-pro', False),
    ],
)
def test_reasoning_effort_parity_with_main(model, expected):
    assert get_features(model).supports_reasoning_effort is expected


def test_prompt_cache_haiku_variants():
    assert get_features('claude-3-5-haiku-20241022').supports_prompt_cache is True
    assert get_features('claude-3.5-haiku-20241022').supports_prompt_cache is True


def test_stop_words_grok_provider_prefixed():
    assert get_features('xai/grok-4-0709').supports_stop_words is False
    assert get_features('grok-4-0709').supports_stop_words is False
    assert get_features('xai/grok-code-fast-1').supports_stop_words is False
    assert get_features('grok-code-fast-1').supports_stop_words is False


@pytest.mark.parametrize(
    'model',
    [
        'o1-mini',
        'o1-2024-12-17',
        'xai/grok-4-0709',
        'xai/grok-code-fast-1',
        'deepseek/DeepSeek-R1-0528:671b-Q4_K_XL',
        'DeepSeek-R1-0528',
    ],
)
def test_supports_stop_words_false_models(model):
    features = get_features(model)
    assert features.supports_stop_words is False
