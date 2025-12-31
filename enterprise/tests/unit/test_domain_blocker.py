"""Unit tests for DomainBlocker class."""

import pytest
from server.auth.domain_blocker import DomainBlocker


@pytest.fixture
def domain_blocker():
    """Create a DomainBlocker instance for testing."""
    return DomainBlocker()


@pytest.mark.parametrize(
    'blocked_domains,expected',
    [
        (['colsch.us', 'other-domain.com'], True),
        (['example.com'], True),
        ([], False),
    ],
)
def test_is_active(domain_blocker, blocked_domains, expected):
    """Test that is_active returns correct value based on blocked domains configuration."""
    # Arrange
    domain_blocker.blocked_domains = blocked_domains

    # Act
    result = domain_blocker.is_active()

    # Assert
    assert result == expected


@pytest.mark.parametrize(
    'email,expected_domain',
    [
        ('user@example.com', 'example.com'),
        ('test@colsch.us', 'colsch.us'),
        ('user.name@other-domain.com', 'other-domain.com'),
        ('USER@EXAMPLE.COM', 'example.com'),  # Case insensitive
        ('user@EXAMPLE.COM', 'example.com'),
        ('  user@example.com  ', 'example.com'),  # Whitespace handling
    ],
)
def test_extract_domain_valid_emails(domain_blocker, email, expected_domain):
    """Test that _extract_domain correctly extracts and normalizes domains from valid emails."""
    # Act
    result = domain_blocker._extract_domain(email)

    # Assert
    assert result == expected_domain


@pytest.mark.parametrize(
    'email,expected',
    [
        (None, None),
        ('', None),
        ('invalid-email', None),
        ('user@', None),  # Empty domain after @
        ('no-at-sign', None),
    ],
)
def test_extract_domain_invalid_emails(domain_blocker, email, expected):
    """Test that _extract_domain returns None for invalid email formats."""
    # Act
    result = domain_blocker._extract_domain(email)

    # Assert
    assert result == expected


def test_is_domain_blocked_when_inactive(domain_blocker):
    """Test that is_domain_blocked returns False when blocking is not active."""
    # Arrange
    domain_blocker.blocked_domains = []

    # Act
    result = domain_blocker.is_domain_blocked('user@colsch.us')

    # Assert
    assert result is False


def test_is_domain_blocked_with_none_email(domain_blocker):
    """Test that is_domain_blocked returns False when email is None."""
    # Arrange
    domain_blocker.blocked_domains = ['colsch.us']

    # Act
    result = domain_blocker.is_domain_blocked(None)

    # Assert
    assert result is False


def test_is_domain_blocked_with_empty_email(domain_blocker):
    """Test that is_domain_blocked returns False when email is empty."""
    # Arrange
    domain_blocker.blocked_domains = ['colsch.us']

    # Act
    result = domain_blocker.is_domain_blocked('')

    # Assert
    assert result is False


def test_is_domain_blocked_with_invalid_email(domain_blocker):
    """Test that is_domain_blocked returns False when email format is invalid."""
    # Arrange
    domain_blocker.blocked_domains = ['colsch.us']

    # Act
    result = domain_blocker.is_domain_blocked('invalid-email')

    # Assert
    assert result is False


def test_is_domain_blocked_domain_not_blocked(domain_blocker):
    """Test that is_domain_blocked returns False when domain is not in blocked list."""
    # Arrange
    domain_blocker.blocked_domains = ['colsch.us', 'other-domain.com']

    # Act
    result = domain_blocker.is_domain_blocked('user@example.com')

    # Assert
    assert result is False


def test_is_domain_blocked_domain_blocked(domain_blocker):
    """Test that is_domain_blocked returns True when domain is in blocked list."""
    # Arrange
    domain_blocker.blocked_domains = ['colsch.us', 'other-domain.com']

    # Act
    result = domain_blocker.is_domain_blocked('user@colsch.us')

    # Assert
    assert result is True


def test_is_domain_blocked_case_insensitive(domain_blocker):
    """Test that is_domain_blocked performs case-insensitive domain matching."""
    # Arrange
    domain_blocker.blocked_domains = ['colsch.us']

    # Act
    result = domain_blocker.is_domain_blocked('user@COLSCH.US')

    # Assert
    assert result is True


def test_is_domain_blocked_multiple_blocked_domains(domain_blocker):
    """Test that is_domain_blocked correctly checks against multiple blocked domains."""
    # Arrange
    domain_blocker.blocked_domains = ['colsch.us', 'other-domain.com', 'blocked.org']

    # Act
    result1 = domain_blocker.is_domain_blocked('user@other-domain.com')
    result2 = domain_blocker.is_domain_blocked('user@blocked.org')
    result3 = domain_blocker.is_domain_blocked('user@allowed.com')

    # Assert
    assert result1 is True
    assert result2 is True
    assert result3 is False


def test_is_domain_blocked_with_whitespace(domain_blocker):
    """Test that is_domain_blocked handles emails with whitespace correctly."""
    # Arrange
    domain_blocker.blocked_domains = ['colsch.us']

    # Act
    result = domain_blocker.is_domain_blocked('  user@colsch.us  ')

    # Assert
    assert result is True
