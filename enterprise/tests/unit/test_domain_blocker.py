"""Unit tests for DomainBlocker class."""

from unittest.mock import MagicMock

import pytest
from server.auth.domain_blocker import DomainBlocker


@pytest.fixture
def mock_store():
    """Create a mock BlockedEmailDomainStore for testing."""
    return MagicMock()


@pytest.fixture
def domain_blocker(mock_store):
    """Create a DomainBlocker instance for testing with a mocked store."""
    return DomainBlocker(store=mock_store)


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


def test_is_domain_blocked_with_none_email(domain_blocker, mock_store):
    """Test that is_domain_blocked returns False when email is None."""
    # Arrange
    mock_store.is_domain_blocked.return_value = True

    # Act
    result = domain_blocker.is_domain_blocked(None)

    # Assert
    assert result is False
    mock_store.is_domain_blocked.assert_not_called()


def test_is_domain_blocked_with_empty_email(domain_blocker, mock_store):
    """Test that is_domain_blocked returns False when email is empty."""
    # Arrange
    mock_store.is_domain_blocked.return_value = True

    # Act
    result = domain_blocker.is_domain_blocked('')

    # Assert
    assert result is False
    mock_store.is_domain_blocked.assert_not_called()


def test_is_domain_blocked_with_invalid_email(domain_blocker, mock_store):
    """Test that is_domain_blocked returns False when email format is invalid."""
    # Arrange
    mock_store.is_domain_blocked.return_value = True

    # Act
    result = domain_blocker.is_domain_blocked('invalid-email')

    # Assert
    assert result is False
    mock_store.is_domain_blocked.assert_not_called()


def test_is_domain_blocked_domain_not_blocked(domain_blocker, mock_store):
    """Test that is_domain_blocked returns False when domain is not blocked."""
    # Arrange
    mock_store.is_domain_blocked.return_value = False

    # Act
    result = domain_blocker.is_domain_blocked('user@example.com')

    # Assert
    assert result is False
    mock_store.is_domain_blocked.assert_called_once_with('example.com')


def test_is_domain_blocked_domain_blocked(domain_blocker, mock_store):
    """Test that is_domain_blocked returns True when domain is blocked."""
    # Arrange
    mock_store.is_domain_blocked.return_value = True

    # Act
    result = domain_blocker.is_domain_blocked('user@colsch.us')

    # Assert
    assert result is True
    mock_store.is_domain_blocked.assert_called_once_with('colsch.us')


def test_is_domain_blocked_case_insensitive(domain_blocker, mock_store):
    """Test that is_domain_blocked performs case-insensitive domain extraction."""
    # Arrange
    mock_store.is_domain_blocked.return_value = True

    # Act
    result = domain_blocker.is_domain_blocked('user@COLSCH.US')

    # Assert
    assert result is True
    mock_store.is_domain_blocked.assert_called_once_with('colsch.us')


def test_is_domain_blocked_with_whitespace(domain_blocker, mock_store):
    """Test that is_domain_blocked handles emails with whitespace correctly."""
    # Arrange
    mock_store.is_domain_blocked.return_value = True

    # Act
    result = domain_blocker.is_domain_blocked('  user@colsch.us  ')

    # Assert
    assert result is True
    mock_store.is_domain_blocked.assert_called_once_with('colsch.us')


def test_is_domain_blocked_multiple_blocked_domains(domain_blocker, mock_store):
    """Test that is_domain_blocked correctly checks multiple domains."""
    # Arrange
    mock_store.is_domain_blocked.side_effect = lambda domain: domain in [
        'other-domain.com',
        'blocked.org',
    ]

    # Act
    result1 = domain_blocker.is_domain_blocked('user@other-domain.com')
    result2 = domain_blocker.is_domain_blocked('user@blocked.org')
    result3 = domain_blocker.is_domain_blocked('user@allowed.com')

    # Assert
    assert result1 is True
    assert result2 is True
    assert result3 is False
    assert mock_store.is_domain_blocked.call_count == 3


def test_is_domain_blocked_tld_pattern_blocks_matching_domain(
    domain_blocker, mock_store
):
    """Test that TLD pattern blocks domains ending with that TLD."""
    # Arrange
    mock_store.is_domain_blocked.return_value = True

    # Act
    result = domain_blocker.is_domain_blocked('user@company.us')

    # Assert
    assert result is True
    mock_store.is_domain_blocked.assert_called_once_with('company.us')


def test_is_domain_blocked_tld_pattern_blocks_subdomain_with_tld(
    domain_blocker, mock_store
):
    """Test that TLD pattern blocks subdomains with that TLD."""
    # Arrange
    mock_store.is_domain_blocked.return_value = True

    # Act
    result = domain_blocker.is_domain_blocked('user@subdomain.company.us')

    # Assert
    assert result is True
    mock_store.is_domain_blocked.assert_called_once_with('subdomain.company.us')


def test_is_domain_blocked_tld_pattern_does_not_block_different_tld(
    domain_blocker, mock_store
):
    """Test that TLD pattern does not block domains with different TLD."""
    # Arrange
    mock_store.is_domain_blocked.return_value = False

    # Act
    result = domain_blocker.is_domain_blocked('user@company.com')

    # Assert
    assert result is False
    mock_store.is_domain_blocked.assert_called_once_with('company.com')


def test_is_domain_blocked_tld_pattern_case_insensitive(domain_blocker, mock_store):
    """Test that TLD pattern matching is case-insensitive."""
    # Arrange
    mock_store.is_domain_blocked.return_value = True

    # Act
    result = domain_blocker.is_domain_blocked('user@COMPANY.US')

    # Assert
    assert result is True
    mock_store.is_domain_blocked.assert_called_once_with('company.us')


def test_is_domain_blocked_tld_pattern_with_multi_level_tld(domain_blocker, mock_store):
    """Test that TLD pattern works with multi-level TLDs like .co.uk."""
    # Arrange
    mock_store.is_domain_blocked.side_effect = lambda domain: domain.endswith('.co.uk')

    # Act
    result_match = domain_blocker.is_domain_blocked('user@example.co.uk')
    result_subdomain = domain_blocker.is_domain_blocked('user@api.example.co.uk')
    result_no_match = domain_blocker.is_domain_blocked('user@example.uk')

    # Assert
    assert result_match is True
    assert result_subdomain is True
    assert result_no_match is False


def test_is_domain_blocked_domain_pattern_blocks_exact_match(
    domain_blocker, mock_store
):
    """Test that domain pattern blocks exact domain match."""
    # Arrange
    mock_store.is_domain_blocked.return_value = True

    # Act
    result = domain_blocker.is_domain_blocked('user@example.com')

    # Assert
    assert result is True
    mock_store.is_domain_blocked.assert_called_once_with('example.com')


def test_is_domain_blocked_domain_pattern_blocks_subdomain(domain_blocker, mock_store):
    """Test that domain pattern blocks subdomains of that domain."""
    # Arrange
    mock_store.is_domain_blocked.return_value = True

    # Act
    result = domain_blocker.is_domain_blocked('user@subdomain.example.com')

    # Assert
    assert result is True
    mock_store.is_domain_blocked.assert_called_once_with('subdomain.example.com')


def test_is_domain_blocked_domain_pattern_blocks_multi_level_subdomain(
    domain_blocker, mock_store
):
    """Test that domain pattern blocks multi-level subdomains."""
    # Arrange
    mock_store.is_domain_blocked.return_value = True

    # Act
    result = domain_blocker.is_domain_blocked('user@api.v2.example.com')

    # Assert
    assert result is True
    mock_store.is_domain_blocked.assert_called_once_with('api.v2.example.com')


def test_is_domain_blocked_domain_pattern_does_not_block_similar_domain(
    domain_blocker, mock_store
):
    """Test that domain pattern does not block domains that contain but don't match the pattern."""
    # Arrange
    mock_store.is_domain_blocked.return_value = False

    # Act
    result = domain_blocker.is_domain_blocked('user@notexample.com')

    # Assert
    assert result is False
    mock_store.is_domain_blocked.assert_called_once_with('notexample.com')


def test_is_domain_blocked_domain_pattern_does_not_block_different_tld(
    domain_blocker, mock_store
):
    """Test that domain pattern does not block same domain with different TLD."""
    # Arrange
    mock_store.is_domain_blocked.return_value = False

    # Act
    result = domain_blocker.is_domain_blocked('user@example.org')

    # Assert
    assert result is False
    mock_store.is_domain_blocked.assert_called_once_with('example.org')


def test_is_domain_blocked_subdomain_pattern_blocks_exact_and_nested(
    domain_blocker, mock_store
):
    """Test that blocking a subdomain also blocks its nested subdomains."""
    # Arrange
    mock_store.is_domain_blocked.side_effect = (
        lambda domain: 'api.example.com' in domain
    )

    # Act
    result_exact = domain_blocker.is_domain_blocked('user@api.example.com')
    result_nested = domain_blocker.is_domain_blocked('user@v1.api.example.com')
    result_parent = domain_blocker.is_domain_blocked('user@example.com')

    # Assert
    assert result_exact is True
    assert result_nested is True
    assert result_parent is False


def test_is_domain_blocked_domain_with_hyphens(domain_blocker, mock_store):
    """Test that domain patterns work with hyphenated domains."""
    # Arrange
    mock_store.is_domain_blocked.return_value = True

    # Act
    result_exact = domain_blocker.is_domain_blocked('user@my-company.com')
    result_subdomain = domain_blocker.is_domain_blocked('user@api.my-company.com')

    # Assert
    assert result_exact is True
    assert result_subdomain is True
    assert mock_store.is_domain_blocked.call_count == 2


def test_is_domain_blocked_domain_with_numbers(domain_blocker, mock_store):
    """Test that domain patterns work with numeric domains."""
    # Arrange
    mock_store.is_domain_blocked.return_value = True

    # Act
    result_exact = domain_blocker.is_domain_blocked('user@test123.com')
    result_subdomain = domain_blocker.is_domain_blocked('user@api.test123.com')

    # Assert
    assert result_exact is True
    assert result_subdomain is True
    assert mock_store.is_domain_blocked.call_count == 2


def test_is_domain_blocked_very_long_subdomain_chain(domain_blocker, mock_store):
    """Test that blocking works with very long subdomain chains."""
    # Arrange
    mock_store.is_domain_blocked.return_value = True

    # Act
    result = domain_blocker.is_domain_blocked(
        'user@level4.level3.level2.level1.example.com'
    )

    # Assert
    assert result is True
    mock_store.is_domain_blocked.assert_called_once_with(
        'level4.level3.level2.level1.example.com'
    )


def test_is_domain_blocked_handles_store_exception(domain_blocker, mock_store):
    """Test that is_domain_blocked returns False when store raises an exception."""
    # Arrange
    mock_store.is_domain_blocked.side_effect = Exception('Database connection error')

    # Act
    result = domain_blocker.is_domain_blocked('user@example.com')

    # Assert
    assert result is False
    mock_store.is_domain_blocked.assert_called_once_with('example.com')
