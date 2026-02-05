"""Tests for openhands.utils.llm module."""

import pytest

from openhands.utils.llm import is_openhands_model


class TestIsOpenhandsModel:
    """Tests for the is_openhands_model function."""

    def test_openhands_model_returns_true(self):
        """Test that models with 'openhands/' prefix return True."""
        assert is_openhands_model('openhands/claude-sonnet-4-5-20250929') is True
        assert is_openhands_model('openhands/gpt-5-2025-08-07') is True
        assert is_openhands_model('openhands/gemini-2.5-pro') is True

    def test_non_openhands_model_returns_false(self):
        """Test that models without 'openhands/' prefix return False."""
        assert is_openhands_model('gpt-4') is False
        assert is_openhands_model('claude-3-opus-20240229') is False
        assert is_openhands_model('anthropic/claude-3-opus-20240229') is False
        assert is_openhands_model('openai/gpt-4') is False

    def test_none_model_returns_false(self):
        """Test that None model returns False."""
        assert is_openhands_model(None) is False

    def test_empty_string_returns_false(self):
        """Test that empty string returns False."""
        assert is_openhands_model('') is False

    def test_similar_prefix_not_matched(self):
        """Test that similar prefixes don't incorrectly match."""
        assert is_openhands_model('openhands') is False  # Missing slash
        assert is_openhands_model('openhandsx/model') is False  # Extra char
        assert is_openhands_model('OPENHANDS/model') is False  # Wrong case
