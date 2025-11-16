#!/usr/bin/env python3
"""
Core Settings Logic tests
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from openhands_cli.user_actions.settings_action import (
    NonEmptyValueValidator,
    SettingsType,
    choose_llm_model,
    choose_llm_provider,
    prompt_api_key,
    settings_type_confirmation,
)
from openhands_cli.tui.settings.settings_screen import _sanitize_model_identifier
from prompt_toolkit.completion import FuzzyWordCompleter
from prompt_toolkit.validation import ValidationError
from pydantic import SecretStr

# -------------------------------
# Settings type selection
# -------------------------------


def test_settings_type_selection(mock_cli_interactions: Any) -> None:
    mocks = mock_cli_interactions

    # Basic
    mocks.cli_confirm.return_value = 0
    assert settings_type_confirmation() == SettingsType.BASIC

    # Cancel/Go back
    mocks.cli_confirm.return_value = 2
    with pytest.raises(KeyboardInterrupt):
        settings_type_confirmation()


# -------------------------------
# Provider selection flows
# -------------------------------


def test_provider_selection_with_predefined_options(
    mock_verified_models: Any, mock_cli_interactions: Any
) -> None:
    from openhands_cli.tui.utils import StepCounter

    mocks = mock_cli_interactions
    # first option among display_options is index 0
    mocks.cli_confirm.return_value = 0
    step_counter = StepCounter(1)
    result = choose_llm_provider(step_counter)
    assert result == 'openai'


def test_provider_selection_with_custom_input(
    mock_verified_models: Any, mock_cli_interactions: Any
) -> None:
    from openhands_cli.tui.utils import StepCounter

    mocks = mock_cli_interactions
    # Due to overlapping provider keys between VERIFIED and UNVERIFIED in fixture,
    # display_options contains 4 providers (with duplicates) + alternate at index 4
    mocks.cli_confirm.return_value = 4
    mocks.cli_text_input.return_value = 'my-provider'
    step_counter = StepCounter(1)
    result = choose_llm_provider(step_counter)
    assert result == 'my-provider'

    # Verify fuzzy completer passed
    _, kwargs = mocks.cli_text_input.call_args
    assert isinstance(kwargs['completer'], FuzzyWordCompleter)


# -------------------------------
# Model selection flows
# -------------------------------


def test_model_selection_flows(
    mock_verified_models: Any, mock_cli_interactions: Any
) -> None:
    from openhands_cli.tui.utils import StepCounter

    mocks = mock_cli_interactions

    # Direct pick from predefined list
    mocks.cli_confirm.return_value = 0
    step_counter = StepCounter(1)
    result = choose_llm_model(step_counter, 'openai')
    assert result in ['gpt-4o']

    # Choose custom model via input
    mocks.cli_confirm.return_value = 4  # for provider with >=4 models this would be alt; in our data openai has 3 -> alt index is 3
    mocks.cli_text_input.return_value = 'custom-model'
    # Adjust to actual alt index produced by code (len(models[:4]) yields 3 + 1 alt -> index 3)
    mocks.cli_confirm.return_value = 3
    step_counter2 = StepCounter(1)
    result2 = choose_llm_model(step_counter2, 'openai')
    assert result2 == 'custom-model'


def test_sanitize_model_identifier_handles_ui_artifacts() -> None:
    assert (
        _sanitize_model_identifier('gemini/2.5 gemini-2.0-flash-lite')
        == 'gemini/gemini-2.0-flash-lite'
    )
    assert (
        _sanitize_model_identifier('(Step 2/3) gemini-2.0-flash-lite')
        == 'gemini-2.0-flash-lite'
    )
    assert (
        _sanitize_model_identifier('openrouter/gemini/gemini-2.5-pro')
        == 'openrouter/gemini/gemini-2.5-pro'
    )


# -------------------------------
# API key validation and prompting
# -------------------------------


def test_api_key_validation_and_prompting(mock_cli_interactions: Any) -> None:
    # Validator standalone
    validator = NonEmptyValueValidator()
    doc = MagicMock()
    doc.text = 'sk-abc'
    validator.validate(doc)

    doc_empty = MagicMock()
    doc_empty.text = ''
    with pytest.raises(ValidationError):
        validator.validate(doc_empty)

    # Prompting for new key enforces validator
    from openhands_cli.tui.utils import StepCounter

    mocks = mock_cli_interactions
    mocks.cli_text_input.return_value = 'sk-new'
    step_counter = StepCounter(1)
    new_key = prompt_api_key(step_counter, 'provider')
    assert new_key == 'sk-new'
    assert mocks.cli_text_input.call_args[1]['validator'] is not None

    # Prompting with existing key shows mask and no validator
    mocks.cli_text_input.reset_mock()
    mocks.cli_text_input.return_value = 'sk-updated'
    existing = SecretStr('sk-existing-123')
    step_counter2 = StepCounter(1)
    updated = prompt_api_key(step_counter2, 'provider', existing)
    assert updated == 'sk-updated'
    assert mocks.cli_text_input.call_args[1]['validator'] is None
    assert 'sk-***' in mocks.cli_text_input.call_args[0][0]
