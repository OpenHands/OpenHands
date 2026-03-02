"""Tests for the resolve_display_name helper.

The resolve_display_name helper extracts the best available display name from
Keycloak user_info claims. It is used by both the /api/user/info fallback path
and the user_store create/migrate paths to avoid duplicating name-resolution logic.

The fallback chain is: name → given_name + family_name → None.
It intentionally does NOT fall back to preferred_username/username — callers
that need a guaranteed non-None value handle that separately, because the
/api/user/info route should return name=None when no real name is available.
"""

from utils.identity import resolve_display_name


class TestResolveDisplayName:
    """Test resolve_display_name with various Keycloak claim combinations."""

    def test_returns_name_when_present(self):
        """When user_info has a 'name' claim, use it directly."""
        user_info = {
            'sub': '123',
            'name': 'Jane Doe',
            'given_name': 'Jane',
            'family_name': 'Doe',
            'preferred_username': 'j.doe',
        }
        assert resolve_display_name(user_info) == 'Jane Doe'

    def test_combines_given_and_family_name_when_name_absent(self):
        """When 'name' is missing, combine given_name + family_name."""
        user_info = {
            'sub': '123',
            'given_name': 'Jane',
            'family_name': 'Doe',
            'preferred_username': 'j.doe',
        }
        assert resolve_display_name(user_info) == 'Jane Doe'

    def test_uses_given_name_only_when_family_name_absent(self):
        """When only given_name is available, use it alone."""
        user_info = {
            'sub': '123',
            'given_name': 'Jane',
            'preferred_username': 'j.doe',
        }
        assert resolve_display_name(user_info) == 'Jane'

    def test_uses_family_name_only_when_given_name_absent(self):
        """When only family_name is available, use it alone."""
        user_info = {
            'sub': '123',
            'family_name': 'Doe',
            'preferred_username': 'j.doe',
        }
        assert resolve_display_name(user_info) == 'Doe'

    def test_returns_none_when_no_name_claims(self):
        """When no name claims exist at all, return None."""
        user_info = {
            'sub': '123',
            'preferred_username': 'j.doe',
            'email': 'jane@example.com',
        }
        assert resolve_display_name(user_info) is None

    def test_returns_none_for_empty_name(self):
        """When 'name' is an empty string, treat it as absent."""
        user_info = {
            'sub': '123',
            'name': '',
            'preferred_username': 'j.doe',
        }
        assert resolve_display_name(user_info) is None

    def test_returns_none_for_whitespace_only_name(self):
        """When 'name' is whitespace only, treat it as absent."""
        user_info = {
            'sub': '123',
            'name': '   ',
            'preferred_username': 'j.doe',
        }
        assert resolve_display_name(user_info) is None

    def test_returns_none_for_empty_given_and_family_names(self):
        """When given_name and family_name are both empty strings, return None."""
        user_info = {
            'sub': '123',
            'given_name': '',
            'family_name': '',
            'preferred_username': 'j.doe',
        }
        assert resolve_display_name(user_info) is None

    def test_returns_none_for_empty_dict(self):
        """An empty user_info dict returns None."""
        assert resolve_display_name({}) is None

    def test_strips_whitespace_from_combined_name(self):
        """Whitespace around given_name/family_name is stripped."""
        user_info = {
            'sub': '123',
            'given_name': '  Jane  ',
            'family_name': '  Doe  ',
        }
        assert resolve_display_name(user_info) == 'Jane Doe'

    def test_name_claim_takes_priority_over_given_family(self):
        """When both 'name' and given/family are present, 'name' wins."""
        user_info = {
            'sub': '123',
            'name': 'Dr. Jane Doe',
            'given_name': 'Jane',
            'family_name': 'Doe',
        }
        assert resolve_display_name(user_info) == 'Dr. Jane Doe'
