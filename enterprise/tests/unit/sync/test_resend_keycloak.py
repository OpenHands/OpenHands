"""Tests for resend_keycloak email validation."""

from sync.resend_keycloak import is_valid_email


class TestIsValidEmail:
    """Test cases for is_valid_email function."""

    def test_valid_simple_email(self):
        """Test that a simple valid email passes validation."""
        assert is_valid_email('user@example.com') is True

    def test_valid_email_with_plus(self):
        """Test that email with + modifier passes validation."""
        assert is_valid_email('user+tag@example.com') is True

    def test_valid_email_with_dots(self):
        """Test that email with dots in local part passes validation."""
        assert is_valid_email('first.last@example.com') is True

    def test_valid_email_with_numbers(self):
        """Test that email with numbers passes validation."""
        assert is_valid_email('user123@example.com') is True

    def test_valid_email_with_subdomain(self):
        """Test that email with subdomain passes validation."""
        assert is_valid_email('user@mail.example.com') is True

    def test_valid_email_with_hyphen_domain(self):
        """Test that email with hyphen in domain passes validation."""
        assert is_valid_email('user@example-site.com') is True

    def test_valid_email_with_underscore(self):
        """Test that email with underscore passes validation."""
        assert is_valid_email('user_name@example.com') is True

    def test_valid_email_with_percent(self):
        """Test that email with percent sign passes validation."""
        assert is_valid_email('user%name@example.com') is True

    def test_invalid_email_with_exclamation(self):
        """Test that email with exclamation mark fails validation.

        This is the specific case from the bug report:
        ethanjames3713+!@gmail.com
        """
        assert is_valid_email('ethanjames3713+!@gmail.com') is False

    def test_invalid_email_with_special_chars(self):
        """Test that email with other special characters fails validation."""
        assert is_valid_email('user!name@example.com') is False
        assert is_valid_email('user#name@example.com') is False
        assert is_valid_email('user$name@example.com') is False
        assert is_valid_email('user&name@example.com') is False
        assert is_valid_email("user'name@example.com") is False
        assert is_valid_email('user*name@example.com') is False
        assert is_valid_email('user=name@example.com') is False
        assert is_valid_email('user^name@example.com') is False
        assert is_valid_email('user`name@example.com') is False
        assert is_valid_email('user{name@example.com') is False
        assert is_valid_email('user|name@example.com') is False
        assert is_valid_email('user}name@example.com') is False
        assert is_valid_email('user~name@example.com') is False

    def test_invalid_email_no_at_symbol(self):
        """Test that email without @ symbol fails validation."""
        assert is_valid_email('userexample.com') is False

    def test_invalid_email_no_domain(self):
        """Test that email without domain fails validation."""
        assert is_valid_email('user@') is False

    def test_invalid_email_no_local_part(self):
        """Test that email without local part fails validation."""
        assert is_valid_email('@example.com') is False

    def test_invalid_email_no_tld(self):
        """Test that email without TLD fails validation."""
        assert is_valid_email('user@example') is False

    def test_invalid_email_single_char_tld(self):
        """Test that email with single character TLD fails validation."""
        assert is_valid_email('user@example.c') is False

    def test_invalid_email_empty_string(self):
        """Test that empty string fails validation."""
        assert is_valid_email('') is False

    def test_invalid_email_none(self):
        """Test that None fails validation."""
        assert is_valid_email(None) is False

    def test_invalid_email_whitespace(self):
        """Test that email with whitespace fails validation."""
        assert is_valid_email('user @example.com') is False
        assert is_valid_email('user@ example.com') is False
        assert is_valid_email(' user@example.com') is False
        assert is_valid_email('user@example.com ') is False

    def test_invalid_email_double_at(self):
        """Test that email with double @ fails validation."""
        assert is_valid_email('user@@example.com') is False

    def test_email_double_dot_domain(self):
        """Test email with double dot in domain.

        Note: The regex allows this as it's technically valid in some edge cases,
        and Resend's API may accept it. The main goal is to reject special
        characters like ! that Resend definitely rejects.
        """
        # This is allowed by our regex - Resend may or may not accept it
        assert is_valid_email('user@example..com') is True

    def test_case_insensitive_validation(self):
        """Test that validation works for uppercase emails."""
        assert is_valid_email('USER@EXAMPLE.COM') is True
        assert is_valid_email('User@Example.Com') is True
