import pathlib

import pytest
from pydantic import ValidationError

from openhands.core.config import OpenHandsConfig
from openhands.core.config.llm_config import LLMConfig
from openhands.core.config.utils import load_from_toml


@pytest.fixture
def default_config(monkeypatch):
    # Fixture to provide a default OpenHandsConfig instance
    yield OpenHandsConfig()


@pytest.fixture
def generic_llm_toml(tmp_path: pathlib.Path) -> str:
    """Fixture to create a generic LLM TOML configuration with all custom LLMs
    providing mandatory 'model' and 'api_key', and testing fallback to the generic section values
    for other attributes like 'num_retries'.
    """
    toml_content = """
[core]
workspace_base = "./workspace"

[llm]
model = "base-model"
api_key = "base-api-key"
num_retries = 3

[llm.custom1]
model = "custom-model-1"
api_key = "custom-api-key-1"
# 'num_retries' is not overridden and should fallback to the value from [llm]

[llm.custom2]
model = "custom-model-2"
api_key = "custom-api-key-2"
num_retries = 5  # Overridden value

[llm.custom3]
model = "custom-model-3"
api_key = "custom-api-key-3"
# No overrides for additional attributes
    """
    toml_file = tmp_path / 'llm_config.toml'
    toml_file.write_text(toml_content)
    return str(toml_file)


def test_load_from_toml_llm_with_fallback(
    default_config: OpenHandsConfig, generic_llm_toml: str
) -> None:
    """Test that custom LLM configurations fallback non-overridden attributes
    like 'num_retries' from the generic [llm] section.
    """
    load_from_toml(default_config, generic_llm_toml)

    # Verify generic LLM configuration
    generic_llm = default_config.get_llm_config('llm')
    assert generic_llm.model == 'base-model'
    assert generic_llm.api_key.get_secret_value() == 'base-api-key'
    assert generic_llm.num_retries == 3

    # Verify custom1 LLM falls back 'num_retries' from base
    custom1 = default_config.get_llm_config('custom1')
    assert custom1.model == 'custom-model-1'
    assert custom1.api_key.get_secret_value() == 'custom-api-key-1'
    assert custom1.num_retries == 3  # from [llm]

    # Verify custom2 LLM overrides 'num_retries'
    custom2 = default_config.get_llm_config('custom2')
    assert custom2.model == 'custom-model-2'
    assert custom2.api_key.get_secret_value() == 'custom-api-key-2'
    assert custom2.num_retries == 5  # overridden value

    # Verify custom3 LLM inherits all attributes except 'model' and 'api_key'
    custom3 = default_config.get_llm_config('custom3')
    assert custom3.model == 'custom-model-3'
    assert custom3.api_key.get_secret_value() == 'custom-api-key-3'
    assert custom3.num_retries == 3  # from [llm]


def test_load_from_toml_llm_custom_overrides_all(
    default_config: OpenHandsConfig, tmp_path: pathlib.Path
) -> None:
    """Test that a custom LLM can fully override all attributes from the generic [llm] section."""
    toml_content = """
[core]
workspace_base = "./workspace"

[llm]
model = "base-model"
api_key = "base-api-key"
num_retries = 3

[llm.custom_full]
model = "full-custom-model"
api_key = "full-custom-api-key"
num_retries = 10
    """
    toml_file = tmp_path / 'full_override_llm.toml'
    toml_file.write_text(toml_content)

    load_from_toml(default_config, str(toml_file))

    # Verify generic LLM configuration remains unchanged
    generic_llm = default_config.get_llm_config('llm')
    assert generic_llm.model == 'base-model'
    assert generic_llm.api_key.get_secret_value() == 'base-api-key'
    assert generic_llm.num_retries == 3

    # Verify custom_full LLM overrides all attributes
    custom_full = default_config.get_llm_config('custom_full')
    assert custom_full.model == 'full-custom-model'
    assert custom_full.api_key.get_secret_value() == 'full-custom-api-key'
    assert custom_full.num_retries == 10  # overridden value


def test_load_from_toml_llm_custom_partial_override(
    default_config: OpenHandsConfig, generic_llm_toml: str
) -> None:
    """Test that custom LLM configurations can partially override attributes
    from the generic [llm] section while inheriting others.
    """
    load_from_toml(default_config, generic_llm_toml)

    # Verify custom1 LLM overrides 'model' and 'api_key' but inherits 'num_retries'
    custom1 = default_config.get_llm_config('custom1')
    assert custom1.model == 'custom-model-1'
    assert custom1.api_key.get_secret_value() == 'custom-api-key-1'
    assert custom1.num_retries == 3  # from [llm]

    # Verify custom2 LLM overrides 'model', 'api_key', and 'num_retries'
    custom2 = default_config.get_llm_config('custom2')
    assert custom2.model == 'custom-model-2'
    assert custom2.api_key.get_secret_value() == 'custom-api-key-2'
    assert custom2.num_retries == 5  # Overridden value


def test_load_from_toml_llm_custom_no_override(
    default_config: OpenHandsConfig, generic_llm_toml: str
) -> None:
    """Test that custom LLM configurations with no additional overrides
    inherit all non-specified attributes from the generic [llm] section.
    """
    load_from_toml(default_config, generic_llm_toml)

    # Verify custom3 LLM inherits 'num_retries' from generic
    custom3 = default_config.get_llm_config('custom3')
    assert custom3.model == 'custom-model-3'
    assert custom3.api_key.get_secret_value() == 'custom-api-key-3'
    assert custom3.num_retries == 3  # from [llm]


def test_load_from_toml_llm_missing_generic(
    default_config: OpenHandsConfig, tmp_path: pathlib.Path
) -> None:
    """Test that custom LLM configurations without a generic [llm] section
    use only their own attributes and fallback to defaults for others.
    """
    toml_content = """
[core]
workspace_base = "./workspace"

[llm.custom_only]
model = "custom-only-model"
api_key = "custom-only-api-key"
    """
    toml_file = tmp_path / 'custom_only_llm.toml'
    toml_file.write_text(toml_content)

    load_from_toml(default_config, str(toml_file))

    # Verify custom_only LLM uses its own attributes and defaults for others
    custom_only = default_config.get_llm_config('custom_only')
    assert custom_only.model == 'custom-only-model'
    assert custom_only.api_key.get_secret_value() == 'custom-only-api-key'
    assert custom_only.num_retries == 5  # default value


def test_load_from_toml_llm_invalid_config(
    default_config: OpenHandsConfig, tmp_path: pathlib.Path
) -> None:
    """Test that invalid custom LLM configurations do not override the generic
    and raise appropriate warnings.
    """
    toml_content = """
[core]
workspace_base = "./workspace"

[llm]
model = "base-model"
api_key = "base-api-key"
num_retries = 3

[llm.invalid_custom]
unknown_attr = "should_not_exist"
    """
    toml_file = tmp_path / 'invalid_custom_llm.toml'
    toml_file.write_text(toml_content)

    load_from_toml(default_config, str(toml_file))

    # Verify generic LLM is loaded correctly
    generic_llm = default_config.get_llm_config('llm')
    assert generic_llm.model == 'base-model'
    assert generic_llm.api_key.get_secret_value() == 'base-api-key'
    assert generic_llm.num_retries == 3

    # Verify invalid_custom LLM does not override generic attributes
    custom_invalid = default_config.get_llm_config('invalid_custom')
    assert custom_invalid.model == 'base-model'
    assert custom_invalid.api_key.get_secret_value() == 'base-api-key'
    assert custom_invalid.num_retries == 3  # default value


def test_azure_model_api_version(
    default_config: OpenHandsConfig, tmp_path: pathlib.Path
) -> None:
    """Test that Azure models get the correct API version by default."""
    toml_content = """
[core]
workspace_base = "./workspace"

[llm]
model = "azure/o3-mini"
api_key = "test-api-key"
    """
    toml_file = tmp_path / 'azure_llm.toml'
    toml_file.write_text(toml_content)

    load_from_toml(default_config, str(toml_file))

    # Verify Azure model gets default API version
    azure_llm = default_config.get_llm_config('llm')
    assert azure_llm.model == 'azure/o3-mini'
    assert azure_llm.api_version == '2024-12-01-preview'

    # Test that non-Azure models don't get default API version
    toml_content = """
[core]
workspace_base = "./workspace"

[llm]
model = "anthropic/claude-3-sonnet"
api_key = "test-api-key"
    """
    toml_file = tmp_path / 'non_azure_llm.toml'
    toml_file.write_text(toml_content)

    load_from_toml(default_config, str(toml_file))

    # Verify non-Azure model doesn't get default API version
    non_azure_llm = default_config.get_llm_config('llm')
    assert non_azure_llm.model == 'anthropic/claude-3-sonnet'
    assert non_azure_llm.api_version is None


class TestBaseUrlValidation:
    """Tests for base_url validation that rejects API endpoint paths."""

    @pytest.mark.parametrize(
        'bad_url',
        [
            'https://my-proxy.com/v1/chat/completions',
            'https://api.openai.com/v1/completions',
            'https://api.anthropic.com/v1/messages',
            'https://my-proxy.com/v1/embeddings',
            'https://my-proxy.com/v1/models',
            'https://my-proxy.com/v1/generations',
            'http://localhost:8080/v1/chat/completions',
            'https://my-proxy.com/v1/chat/completions/',
        ],
    )
    def test_rejects_endpoint_paths(self, bad_url: str) -> None:
        with pytest.raises(
            ValidationError, match='must not include the API endpoint path'
        ):
            LLMConfig(base_url=bad_url)

    @pytest.mark.parametrize(
        'good_url',
        [
            'https://my-proxy.com/v1',
            'https://api.openai.com/v1',
            'https://api.anthropic.com',
            'http://localhost:8080',
            'http://localhost:11434',
            'https://my-company.com/llm-proxy',
            None,
        ],
    )
    def test_accepts_valid_base_urls(self, good_url: str | None) -> None:
        config = LLMConfig(base_url=good_url)
        assert config.base_url == good_url

    def test_toml_with_bad_base_url_is_skipped(
        self, default_config: OpenHandsConfig, tmp_path: pathlib.Path
    ) -> None:
        """A custom LLM section with an invalid base_url should be skipped gracefully."""
        toml_content = """
[core]
workspace_base = "./workspace"

[llm]
model = "base-model"
api_key = "base-api-key"

[llm.bad_base]
model = "custom-model"
api_key = "custom-api-key"
base_url = "https://my-proxy.com/v1/chat/completions"
        """
        toml_file = tmp_path / 'bad_base_url_llm.toml'
        toml_file.write_text(toml_content)

        load_from_toml(default_config, str(toml_file))

        # The base config should still load correctly
        generic_llm = default_config.get_llm_config('llm')
        assert generic_llm.model == 'base-model'

        # The invalid custom section should be skipped — accessing it falls
        # back to the generic config
        bad_base = default_config.get_llm_config('bad_base')
        assert bad_base.model == 'base-model'
