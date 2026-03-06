"""Email validation utilities for preventing duplicate signups with + modifier."""

import re


def extract_base_email(email: str) -> str | None:
    """Extract base email from an email address.

    For emails with + modifier, extracts the base email (local part before + and @, plus domain).
    For emails without + modifier, returns the email as-is.

    Examples:
        extract_base_email("joe+test@example.com") -> "joe@example.com"
        extract_base_email("joe@example.com") -> "joe@example.com"
        extract_base_email("joe+openhands+test@example.com") -> "joe@example.com"

    Args:
        email: The email address to process

    Returns:
        The base email address, or None if email format is invalid
    """
    if not email or '@' not in email:
        return None

    try:
        local_part, domain = email.rsplit('@', 1)
        # Extract the part before + if it exists
        base_local = local_part.split('+', 1)[0]
        return f'{base_local}@{domain}'
    except (ValueError, AttributeError):
        return None


def has_plus_modifier(email: str) -> bool:
    """Check if an email address contains a + modifier.

    Args:
        email: The email address to check

    Returns:
        True if email contains + before @, False otherwise
    """
    if not email or '@' not in email:
        return False

    try:
        local_part, _ = email.rsplit('@', 1)
        return '+' in local_part
    except (ValueError, AttributeError):
        return False


def matches_base_email(email: str, base_email: str) -> bool:
    """Check if an email matches a base email pattern.

    An email matches if:
    - It is exactly the base email (e.g., joe@example.com)
    - It has the same base local part and domain, with or without + modifier
      (e.g., joe+test@example.com matches base joe@example.com)

    Args:
        email: The email address to check
        base_email: The base email to match against

    Returns:
        True if email matches the base pattern, False otherwise
    """
    if not email or not base_email:
        return False

    # Extract base from both emails for comparison
    email_base = extract_base_email(email)
    base_email_normalized = extract_base_email(base_email)

    if not email_base or not base_email_normalized:
        return False

    # Emails match if they have the same base
    return email_base.lower() == base_email_normalized.lower()


def get_base_email_regex_pattern(base_email: str) -> re.Pattern | None:
    """Generate a regex pattern to match emails with the same base.

    For base_email "joe@example.com", the pattern will match:
    - joe@example.com
    - joe+anything@example.com

    Args:
        base_email: The base email address

    Returns:
        A compiled regex pattern, or None if base_email is invalid
    """
    base = extract_base_email(base_email)
    if not base:
        return None

    try:
        local_part, domain = base.rsplit('@', 1)
        # Escape special regex characters in local part and domain
        escaped_local = re.escape(local_part)
        escaped_domain = re.escape(domain)
        # Pattern: joe@example.com OR joe+anything@example.com
        pattern = rf'^{escaped_local}(\+[^@\s]+)?@{escaped_domain}$'
        return re.compile(pattern, re.IGNORECASE)
    except (ValueError, AttributeError):
        return None
