"""Tests for email validation utilities."""

import re

from server.auth.email_validation import (
    extract_base_email,
    get_base_email_regex_pattern,
    has_plus_modifier,
    matches_base_email,
)


class TestExtractBaseEmail:
    """Test cases for extract_base_email function."""

    def test_extract_base_email_with_plus_modifier(self):
        """Test extracting base email from email with + modifier."""
        # Arrange
        email = 'joe+test@example.com'

        # Act
        result = extract_base_email(email)

        # Assert
        assert result == 'joe@example.com'

    def test_extract_base_email_without_plus_modifier(self):
        """Test that email without + modifier is returned as-is."""
        # Arrange
        email = 'joe@example.com'

        # Act
        result = extract_base_email(email)

        # Assert
        assert result == 'joe@example.com'

    def test_extract_base_email_multiple_plus_signs(self):
        """Test extracting base email when multiple + signs exist."""
        # Arrange
        email = 'joe+openhands+test@example.com'

        # Act
        result = extract_base_email(email)

        # Assert
        assert result == 'joe@example.com'

    def test_extract_base_email_invalid_no_at_symbol(self):
        """Test that invalid email without @ returns None."""
        # Arrange
        email = 'invalid-email'

        # Act
        result = extract_base_email(email)

        # Assert
        assert result is None

    def test_extract_base_email_empty_string(self):
        """Test that empty string returns None."""
        # Arrange
        email = ''

        # Act
        result = extract_base_email(email)

        # Assert
        assert result is None

    def test_extract_base_email_none(self):
        """Test that None input returns None."""
        # Arrange
        email = None

        # Act
        result = extract_base_email(email)

        # Assert
        assert result is None


class TestHasPlusModifier:
    """Test cases for has_plus_modifier function."""

    def test_has_plus_modifier_true(self):
        """Test detecting + modifier in email."""
        # Arrange
        email = 'joe+test@example.com'

        # Act
        result = has_plus_modifier(email)

        # Assert
        assert result is True

    def test_has_plus_modifier_false(self):
        """Test that email without + modifier returns False."""
        # Arrange
        email = 'joe@example.com'

        # Act
        result = has_plus_modifier(email)

        # Assert
        assert result is False

    def test_has_plus_modifier_invalid_no_at_symbol(self):
        """Test that invalid email without @ returns False."""
        # Arrange
        email = 'invalid-email'

        # Act
        result = has_plus_modifier(email)

        # Assert
        assert result is False

    def test_has_plus_modifier_empty_string(self):
        """Test that empty string returns False."""
        # Arrange
        email = ''

        # Act
        result = has_plus_modifier(email)

        # Assert
        assert result is False


class TestMatchesBaseEmail:
    """Test cases for matches_base_email function."""

    def test_matches_base_email_exact_match(self):
        """Test that exact base email matches."""
        # Arrange
        email = 'joe@example.com'
        base_email = 'joe@example.com'

        # Act
        result = matches_base_email(email, base_email)

        # Assert
        assert result is True

    def test_matches_base_email_with_plus_variant(self):
        """Test that email with + variant matches base email."""
        # Arrange
        email = 'joe+test@example.com'
        base_email = 'joe@example.com'

        # Act
        result = matches_base_email(email, base_email)

        # Assert
        assert result is True

    def test_matches_base_email_different_base(self):
        """Test that different base emails do not match."""
        # Arrange
        email = 'jane@example.com'
        base_email = 'joe@example.com'

        # Act
        result = matches_base_email(email, base_email)

        # Assert
        assert result is False

    def test_matches_base_email_different_domain(self):
        """Test that same local part but different domain does not match."""
        # Arrange
        email = 'joe@other.com'
        base_email = 'joe@example.com'

        # Act
        result = matches_base_email(email, base_email)

        # Assert
        assert result is False

    def test_matches_base_email_case_insensitive(self):
        """Test that matching is case-insensitive."""
        # Arrange
        email = 'JOE+TEST@EXAMPLE.COM'
        base_email = 'joe@example.com'

        # Act
        result = matches_base_email(email, base_email)

        # Assert
        assert result is True

    def test_matches_base_email_empty_strings(self):
        """Test that empty strings return False."""
        # Arrange
        email = ''
        base_email = 'joe@example.com'

        # Act
        result = matches_base_email(email, base_email)

        # Assert
        assert result is False


class TestGetBaseEmailRegexPattern:
    """Test cases for get_base_email_regex_pattern function."""

    def test_get_base_email_regex_pattern_valid(self):
        """Test generating valid regex pattern for base email."""
        # Arrange
        base_email = 'joe@example.com'

        # Act
        pattern = get_base_email_regex_pattern(base_email)

        # Assert
        assert pattern is not None
        assert isinstance(pattern, re.Pattern)
        assert pattern.match('joe@example.com') is not None
        assert pattern.match('joe+test@example.com') is not None
        assert pattern.match('joe+openhands@example.com') is not None

    def test_get_base_email_regex_pattern_matches_plus_variant(self):
        """Test that regex pattern matches + variant."""
        # Arrange
        base_email = 'joe@example.com'
        pattern = get_base_email_regex_pattern(base_email)

        # Act
        match = pattern.match('joe+test@example.com')

        # Assert
        assert match is not None

    def test_get_base_email_regex_pattern_rejects_different_base(self):
        """Test that regex pattern rejects different base email."""
        # Arrange
        base_email = 'joe@example.com'
        pattern = get_base_email_regex_pattern(base_email)

        # Act
        match = pattern.match('jane@example.com')

        # Assert
        assert match is None

    def test_get_base_email_regex_pattern_rejects_different_domain(self):
        """Test that regex pattern rejects different domain."""
        # Arrange
        base_email = 'joe@example.com'
        pattern = get_base_email_regex_pattern(base_email)

        # Act
        match = pattern.match('joe@other.com')

        # Assert
        assert match is None

    def test_get_base_email_regex_pattern_case_insensitive(self):
        """Test that regex pattern is case-insensitive."""
        # Arrange
        base_email = 'joe@example.com'
        pattern = get_base_email_regex_pattern(base_email)

        # Act
        match = pattern.match('JOE+TEST@EXAMPLE.COM')

        # Assert
        assert match is not None

    def test_get_base_email_regex_pattern_special_characters(self):
        """Test that regex pattern handles special characters in email."""
        # Arrange
        base_email = 'user.name+tag@example-site.com'
        pattern = get_base_email_regex_pattern(base_email)

        # Act
        match = pattern.match('user.name+test@example-site.com')

        # Assert
        assert match is not None

    def test_get_base_email_regex_pattern_invalid_base_email(self):
        """Test that invalid base email returns None."""
        # Arrange
        base_email = 'invalid-email'

        # Act
        pattern = get_base_email_regex_pattern(base_email)

        # Assert
        assert pattern is None
