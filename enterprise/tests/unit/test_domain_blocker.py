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


# ============================================================================
# TLD Blocking Tests (patterns starting with '.')
# ============================================================================


def test_is_domain_blocked_tld_pattern_blocks_matching_domain(domain_blocker):
    """Test that TLD pattern blocks domains ending with that TLD."""
    # Arrange
    domain_blocker.blocked_domains = ['.us']

    # Act
    result = domain_blocker.is_domain_blocked('user@company.us')

    # Assert
    assert result is True


def test_is_domain_blocked_tld_pattern_blocks_subdomain_with_tld(domain_blocker):
    """Test that TLD pattern blocks subdomains with that TLD."""
    # Arrange
    domain_blocker.blocked_domains = ['.us']

    # Act
    result = domain_blocker.is_domain_blocked('user@subdomain.company.us')

    # Assert
    assert result is True


def test_is_domain_blocked_tld_pattern_does_not_block_different_tld(domain_blocker):
    """Test that TLD pattern does not block domains with different TLD."""
    # Arrange
    domain_blocker.blocked_domains = ['.us']

    # Act
    result = domain_blocker.is_domain_blocked('user@company.com')

    # Assert
    assert result is False


def test_is_domain_blocked_tld_pattern_does_not_block_substring_match(
    domain_blocker,
):
    """Test that TLD pattern does not block domains that contain but don't end with the TLD."""
    # Arrange
    domain_blocker.blocked_domains = ['.us']

    # Act
    result = domain_blocker.is_domain_blocked('user@focus.com')

    # Assert
    assert result is False


def test_is_domain_blocked_tld_pattern_case_insensitive(domain_blocker):
    """Test that TLD pattern matching is case-insensitive."""
    # Arrange
    domain_blocker.blocked_domains = ['.us']

    # Act
    result = domain_blocker.is_domain_blocked('user@COMPANY.US')

    # Assert
    assert result is True


def test_is_domain_blocked_multiple_tld_patterns(domain_blocker):
    """Test blocking with multiple TLD patterns."""
    # Arrange
    domain_blocker.blocked_domains = ['.us', '.vn', '.com']

    # Act
    result_us = domain_blocker.is_domain_blocked('user@test.us')
    result_vn = domain_blocker.is_domain_blocked('user@test.vn')
    result_com = domain_blocker.is_domain_blocked('user@test.com')
    result_org = domain_blocker.is_domain_blocked('user@test.org')

    # Assert
    assert result_us is True
    assert result_vn is True
    assert result_com is True
    assert result_org is False


def test_is_domain_blocked_tld_pattern_with_multi_level_tld(domain_blocker):
    """Test that TLD pattern works with multi-level TLDs like .co.uk."""
    # Arrange
    domain_blocker.blocked_domains = ['.co.uk']

    # Act
    result_match = domain_blocker.is_domain_blocked('user@example.co.uk')
    result_subdomain = domain_blocker.is_domain_blocked('user@api.example.co.uk')
    result_no_match = domain_blocker.is_domain_blocked('user@example.uk')

    # Assert
    assert result_match is True
    assert result_subdomain is True
    assert result_no_match is False


# ============================================================================
# Subdomain Blocking Tests (domain patterns now block subdomains)
# ============================================================================


def test_is_domain_blocked_domain_pattern_blocks_exact_match(domain_blocker):
    """Test that domain pattern blocks exact domain match."""
    # Arrange
    domain_blocker.blocked_domains = ['example.com']

    # Act
    result = domain_blocker.is_domain_blocked('user@example.com')

    # Assert
    assert result is True


def test_is_domain_blocked_domain_pattern_blocks_subdomain(domain_blocker):
    """Test that domain pattern blocks subdomains of that domain."""
    # Arrange
    domain_blocker.blocked_domains = ['example.com']

    # Act
    result = domain_blocker.is_domain_blocked('user@subdomain.example.com')

    # Assert
    assert result is True


def test_is_domain_blocked_domain_pattern_blocks_multi_level_subdomain(
    domain_blocker,
):
    """Test that domain pattern blocks multi-level subdomains."""
    # Arrange
    domain_blocker.blocked_domains = ['example.com']

    # Act
    result = domain_blocker.is_domain_blocked('user@api.v2.example.com')

    # Assert
    assert result is True


def test_is_domain_blocked_domain_pattern_does_not_block_similar_domain(
    domain_blocker,
):
    """Test that domain pattern does not block domains that contain but don't match the pattern."""
    # Arrange
    domain_blocker.blocked_domains = ['example.com']

    # Act
    result = domain_blocker.is_domain_blocked('user@notexample.com')

    # Assert
    assert result is False


def test_is_domain_blocked_domain_pattern_does_not_block_different_tld(
    domain_blocker,
):
    """Test that domain pattern does not block same domain with different TLD."""
    # Arrange
    domain_blocker.blocked_domains = ['example.com']

    # Act
    result = domain_blocker.is_domain_blocked('user@example.org')

    # Assert
    assert result is False


def test_is_domain_blocked_subdomain_pattern_blocks_exact_and_nested(domain_blocker):
    """Test that blocking a subdomain also blocks its nested subdomains."""
    # Arrange
    domain_blocker.blocked_domains = ['api.example.com']

    # Act
    result_exact = domain_blocker.is_domain_blocked('user@api.example.com')
    result_nested = domain_blocker.is_domain_blocked('user@v1.api.example.com')
    result_parent = domain_blocker.is_domain_blocked('user@example.com')

    # Assert
    assert result_exact is True
    assert result_nested is True
    assert result_parent is False


# ============================================================================
# Mixed Pattern Tests (TLD + domain patterns together)
# ============================================================================


def test_is_domain_blocked_mixed_patterns_tld_and_domain(domain_blocker):
    """Test blocking with both TLD and domain patterns."""
    # Arrange
    domain_blocker.blocked_domains = ['.us', 'openhands.dev']

    # Act
    result_tld = domain_blocker.is_domain_blocked('user@company.us')
    result_domain = domain_blocker.is_domain_blocked('user@openhands.dev')
    result_subdomain = domain_blocker.is_domain_blocked('user@api.openhands.dev')
    result_allowed = domain_blocker.is_domain_blocked('user@example.com')

    # Assert
    assert result_tld is True
    assert result_domain is True
    assert result_subdomain is True
    assert result_allowed is False


def test_is_domain_blocked_overlapping_patterns(domain_blocker):
    """Test that overlapping patterns (TLD and specific domain) both work."""
    # Arrange
    domain_blocker.blocked_domains = ['.us', 'test.us']

    # Act
    result_specific = domain_blocker.is_domain_blocked('user@test.us')
    result_other_us = domain_blocker.is_domain_blocked('user@other.us')

    # Assert
    assert result_specific is True
    assert result_other_us is True


def test_is_domain_blocked_complex_multi_pattern_scenario(domain_blocker):
    """Test complex scenario with multiple TLD and domain patterns."""
    # Arrange
    domain_blocker.blocked_domains = [
        '.us',
        '.vn',
        'test.com',
        'openhands.dev',
    ]

    # Act & Assert
    # TLD patterns
    assert domain_blocker.is_domain_blocked('user@anything.us') is True
    assert domain_blocker.is_domain_blocked('user@company.vn') is True

    # Domain patterns (exact)
    assert domain_blocker.is_domain_blocked('user@test.com') is True
    assert domain_blocker.is_domain_blocked('user@openhands.dev') is True

    # Domain patterns (subdomains)
    assert domain_blocker.is_domain_blocked('user@api.test.com') is True
    assert domain_blocker.is_domain_blocked('user@staging.openhands.dev') is True

    # Not blocked
    assert domain_blocker.is_domain_blocked('user@allowed.com') is False
    assert domain_blocker.is_domain_blocked('user@example.org') is False


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_is_domain_blocked_domain_with_hyphens(domain_blocker):
    """Test that domain patterns work with hyphenated domains."""
    # Arrange
    domain_blocker.blocked_domains = ['my-company.com']

    # Act
    result_exact = domain_blocker.is_domain_blocked('user@my-company.com')
    result_subdomain = domain_blocker.is_domain_blocked('user@api.my-company.com')

    # Assert
    assert result_exact is True
    assert result_subdomain is True


def test_is_domain_blocked_domain_with_numbers(domain_blocker):
    """Test that domain patterns work with numeric domains."""
    # Arrange
    domain_blocker.blocked_domains = ['test123.com']

    # Act
    result_exact = domain_blocker.is_domain_blocked('user@test123.com')
    result_subdomain = domain_blocker.is_domain_blocked('user@api.test123.com')

    # Assert
    assert result_exact is True
    assert result_subdomain is True


def test_is_domain_blocked_short_tld(domain_blocker):
    """Test that short TLD patterns work correctly."""
    # Arrange
    domain_blocker.blocked_domains = ['.io']

    # Act
    result = domain_blocker.is_domain_blocked('user@company.io')

    # Assert
    assert result is True


def test_is_domain_blocked_very_long_subdomain_chain(domain_blocker):
    """Test that blocking works with very long subdomain chains."""
    # Arrange
    domain_blocker.blocked_domains = ['example.com']

    # Act
    result = domain_blocker.is_domain_blocked(
        'user@level4.level3.level2.level1.example.com'
    )

    # Assert
    assert result is True
