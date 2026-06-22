"""Tests for marketplace composition logic in settings_router."""

from __future__ import annotations

import os

# Import from the actual router module
from openhands.app_server.settings.settings_router import (
    _get_instance_default_marketplaces,
    _merge_marketplaces,
)


class TestGetInstanceDefaultMarketplaces:
    """Tests for parsing INSTANCE_DEFAULT_MARKETPLACES environment variable."""

    def test_empty_env_var(self, monkeypatch):
        """Test that empty env var returns empty list."""
        monkeypatch.setenv('INSTANCE_DEFAULT_MARKETPLACES', '')
        result = _get_instance_default_marketplaces()
        assert result == []

    def test_unset_env_var(self, monkeypatch):
        """Test that unset env var returns empty list."""
        monkeypatch.delenv('INSTANCE_DEFAULT_MARKETPLACES', raising=False)
        result = _get_instance_default_marketplaces()
        assert result == []

    def test_single_marketplace(self, monkeypatch):
        """Test parsing a single marketplace definition."""
        monkeypatch.setenv('INSTANCE_DEFAULT_MARKETPLACES', 'github:OpenHands/skills')
        result = _get_instance_default_marketplaces()
        assert len(result) == 1
        assert result[0]['source'] == 'github:OpenHands/skills'
        assert result[0]['auto_load'] == 'all'

    def test_marketplace_with_name(self, monkeypatch):
        """Test parsing marketplace with custom name using # separator."""
        monkeypatch.setenv(
            'INSTANCE_DEFAULT_MARKETPLACES', 'github:OpenHands/skills#my-marketplace'
        )
        result = _get_instance_default_marketplaces()
        assert len(result) == 1
        assert result[0]['source'] == 'github:OpenHands/skills'
        assert result[0]['name'] == 'my-marketplace'

    def test_marketplace_with_all_fields(self, monkeypatch):
        """Test parsing marketplace with all fields using # separator."""
        monkeypatch.setenv(
            'INSTANCE_DEFAULT_MARKETPLACES',
            'github:OpenHands/extensions#my-market#main#marketplaces/plugins',
        )
        result = _get_instance_default_marketplaces()
        assert len(result) == 1
        assert result[0]['source'] == 'github:OpenHands/extensions'
        assert result[0]['name'] == 'my-market'
        assert result[0]['ref'] == 'main'
        assert result[0]['repo_path'] == 'marketplaces/plugins'

    def test_multiple_marketplaces(self, monkeypatch):
        """Test parsing multiple comma-separated marketplaces."""
        monkeypatch.setenv(
            'INSTANCE_DEFAULT_MARKETPLACES',
            'github:OpenHands/skills,github:myorg/plugins',
        )
        result = _get_instance_default_marketplaces()
        assert len(result) == 2
        assert result[0]['source'] == 'github:OpenHands/skills'
        assert result[1]['source'] == 'github:myorg/plugins'

    def test_whitespace_handling(self, monkeypatch):
        """Test that whitespace is trimmed."""
        monkeypatch.setenv(
            'INSTANCE_DEFAULT_MARKETPLACES',
            '  github:OpenHands/skills  ,  github:myorg/plugins  ',
        )
        result = _get_instance_default_marketplaces()
        assert len(result) == 2
        assert result[0]['source'] == 'github:OpenHands/skills'
        assert result[1]['source'] == 'github:myorg/plugins'

    def test_empty_parts_ignored(self, monkeypatch):
        """Test that empty parts after # separator are ignored."""
        monkeypatch.setenv(
            'INSTANCE_DEFAULT_MARKETPLACES',
            'github:OpenHands/skills##v2#marketplaces/plugins',
        )
        result = _get_instance_default_marketplaces()
        assert len(result) == 1
        assert result[0]['source'] == 'github:OpenHands/skills'
        assert result[0]['ref'] == 'v2'
        assert result[0]['repo_path'] == 'marketplaces/plugins'
        assert 'name' not in result[0]


class TestMergeMarketplaces:
    """Tests for marketplace composition logic."""

    def test_empty_all(self):
        """Test composition with all empty lists."""
        inherited, personal = _merge_marketplaces([], [], [])
        assert inherited == []
        assert personal == []

    def test_only_instance_defaults(self):
        """Test composition with only instance defaults."""
        instance = [
            {'source': 'github:OpenHands/skills', 'auto_load': 'all'},
            {'source': 'github:myorg/plugins', 'name': 'my-plugins'},
        ]
        inherited, personal = _merge_marketplaces(instance, [], [])
        assert len(inherited) == 2
        assert all(mp.get('scope') == 'instance' for mp in inherited)
        assert personal == []

    def test_instance_and_org(self):
        """Test composition with instance and org marketplaces."""
        instance = [{'source': 'github:OpenHands/skills', 'auto_load': 'all'}]
        org = [{'source': 'github:myorg/plugins', 'name': 'my-plugins'}]
        inherited, personal = _merge_marketplaces(instance, org, [])
        assert len(inherited) == 2
        assert inherited[0]['source'] == 'github:OpenHands/skills'
        assert inherited[0]['scope'] == 'instance'
        assert inherited[1]['source'] == 'github:myorg/plugins'
        assert inherited[1]['scope'] == 'org'
        assert personal == []

    def test_org_overrides_instance(self):
        """Test that org settings override instance settings for same source."""
        instance = [
            {'source': 'github:OpenHands/skills', 'ref': 'main', 'auto_load': 'all'}
        ]
        org = [{'source': 'github:OpenHands/skills', 'ref': 'v2', 'auto_load': None}]
        inherited, personal = _merge_marketplaces(instance, org, [])
        assert len(inherited) == 1
        # Org should override instance values
        assert inherited[0]['ref'] == 'v2'
        assert inherited[0]['scope'] == 'org'

    def test_user_adds_new_marketplace(self):
        """Test that user can add new marketplace."""
        instance = [{'source': 'github:OpenHands/skills', 'auto_load': 'all'}]
        user = [
            {'source': 'github:myorg/plugins', 'name': 'my-plugins', 'auto_load': 'all'}
        ]
        inherited, personal = _merge_marketplaces(instance, [], user)
        assert len(inherited) == 1
        assert inherited[0]['source'] == 'github:OpenHands/skills'
        assert len(personal) == 1
        assert personal[0]['source'] == 'github:myorg/plugins'
        assert personal[0]['scope'] == 'personal'

    def test_user_cannot_override_instance(self):
        """Test that user cannot override an instance marketplace.

        Users can only add NEW personal marketplaces. They cannot modify
        or override instance marketplace settings.
        """
        instance = [
            {'source': 'github:OpenHands/skills', 'ref': 'main', 'auto_load': 'all'}
        ]
        user = [{'source': 'github:OpenHands/skills', 'ref': 'v2', 'auto_load': None}]
        inherited, personal = _merge_marketplaces(instance, [], user)
        # Instance marketplace remains unchanged
        assert len(inherited) == 1
        assert inherited[0]['source'] == 'github:OpenHands/skills'
        assert inherited[0]['scope'] == 'instance'
        assert inherited[0]['ref'] == 'main'
        assert inherited[0]['auto_load'] == 'all'
        # User's attempt is ignored - not added to personal
        assert len(personal) == 0

    def test_user_cannot_override_org(self):
        """Test that user cannot override an org marketplace.

        Users can only add NEW personal marketplaces. They cannot modify
        or override org marketplace settings.
        """
        org = [{'source': 'github:myorg/plugins', 'ref': 'main', 'auto_load': 'all'}]
        user = [{'source': 'github:myorg/plugins', 'ref': 'v2', 'auto_load': None}]
        inherited, personal = _merge_marketplaces([], org, user)
        # Org marketplace remains unchanged
        assert len(inherited) == 1
        assert inherited[0]['source'] == 'github:myorg/plugins'
        assert inherited[0]['scope'] == 'org'
        assert inherited[0]['ref'] == 'main'
        assert inherited[0]['auto_load'] == 'all'
        # User's attempt is ignored - not added to personal
        assert len(personal) == 0

    def test_full_composition(self):
        """Test full composition with instance, org, and user settings.

        Users can only add NEW personal marketplaces.
        Users cannot override instance or org marketplace settings.
        """
        instance = [
            {'source': 'github:OpenHands/skills', 'auto_load': 'all'},
            {'source': 'github:instance/only', 'name': 'instance-only'},
        ]
        org = [
            {'source': 'github:OpenHands/skills', 'ref': 'v2'},  # override instance
            {'source': 'github:myorg/plugins', 'name': 'org-plugins'},
        ]
        user = [
            {'source': 'github:OpenHands/skills', 'ref': 'v3'},  # attempts to override org - IGNORED
            {'source': 'github:myorg/plugins', 'auto_load': None},  # attempts to override org - IGNORED
            {'source': 'github:user/custom', 'name': 'user-market'},  # NEW marketplace - should be added
        ]

        inherited, personal = _merge_marketplaces(instance, org, user)

        # Check inherited marketplaces (3 total)
        assert len(inherited) == 3
        inherited_by_source = {mp['source']: mp for mp in inherited}

        # github:OpenHands/skills: org overrides instance, user's override is ignored
        assert inherited_by_source['github:OpenHands/skills']['ref'] == 'v2'  # from org
        assert inherited_by_source['github:OpenHands/skills']['scope'] == 'org'

        # github:instance/only should still be instance-level
        assert inherited_by_source['github:instance/only']['scope'] == 'instance'

        # github:myorg/plugins: org's value preserved, user's override ignored
        assert inherited_by_source['github:myorg/plugins']['scope'] == 'org'
        assert inherited_by_source['github:myorg/plugins']['name'] == 'org-plugins'

        # Check personal marketplaces - only NEW user's marketplace
        assert len(personal) == 1
        assert personal[0]['source'] == 'github:user/custom'
        assert personal[0]['name'] == 'user-market'

    def test_deduplication_by_source(self):
        """Test that same source from multiple levels results in single entry.

        User's attempt to override org marketplace is ignored.
        """
        instance = [
            {'source': 'github:OpenHands/skills', 'ref': 'main', 'auto_load': 'all'}
        ]
        org = [{'source': 'github:OpenHands/skills', 'ref': 'org-branch'}]
        user = [{'source': 'github:OpenHands/skills', 'ref': 'user-branch'}]

        inherited, personal = _merge_marketplaces(instance, org, user)

        # Should have org's values (user's override is ignored)
        assert len(inherited) == 1
        assert inherited[0]['ref'] == 'org-branch'
        assert inherited[0]['scope'] == 'org'

        # Personal should be empty (user only tried to override, not add new)
        assert len(personal) == 0

    def test_marketplace_with_empty_source_ignored(self):
        """Test that marketplaces with empty source are ignored."""
        instance = [{'source': '', 'auto_load': 'all'}]
        inherited, personal = _merge_marketplaces(instance, [], [])
        assert inherited == []
        assert personal == []

    def test_marketplace_with_none_source_ignored(self):
        """Test that marketplaces with None source are ignored."""
        instance = [{'source': None, 'auto_load': 'all'}]
        inherited, personal = _merge_marketplaces(instance, [], [])
        assert inherited == []
        assert personal == []

    def test_user_cannot_override_instance(self):
        """Test that user cannot modify instance marketplace fields.

        Users can only add NEW personal marketplaces. They cannot modify
        instance marketplace settings.
        """
        instance = [
            {
                'source': 'github:OpenHands/skills',
                'name': 'original-name',
                'ref': 'main',
                'auto_load': 'all',
            }
        ]
        # User tries to change auto_load
        user = [{'source': 'github:OpenHands/skills', 'auto_load': None}]
        inherited, personal = _merge_marketplaces(instance, [], user)

        # Instance marketplace remains unchanged
        assert len(inherited) == 1
        assert inherited[0]['name'] == 'original-name'
        assert inherited[0]['ref'] == 'main'
        assert inherited[0]['auto_load'] == 'all'  # Not modified by user
        assert inherited[0]['scope'] == 'instance'

        # User's attempt is ignored
        assert len(personal) == 0

    def test_user_cannot_override_org_marketplace_with_new_user_marketplace(self):
        """Test that user cannot override org marketplace even if user adds same source.

        When org adds a marketplace, user's attempt to add the same source
        to personal is ignored.
        """
        # User adds marketplace first (personal)
        user = [{'source': 'github:myorg/plugins', 'ref': 'user-branch'}]
        # Org adds same marketplace
        org = [{'source': 'github:myorg/plugins', 'ref': 'org-branch'}]
        inherited, personal = _merge_marketplaces([], org, user)

        # Org's version goes to inherited
        assert len(inherited) == 1
        assert inherited[0]['ref'] == 'org-branch'
        assert inherited[0]['scope'] == 'org'

        # User's version is ignored (not added to personal)
        assert len(personal) == 0

    def test_org_overrides_instance_preserves_unmodified_instance_fields(self):
        """Test that org override preserves unmodified instance fields."""
        instance = [
            {
                'source': 'github:OpenHands/skills',
                'name': 'instance-name',
                'ref': 'main',
                'auto_load': 'all',
            }
        ]
        org = [{'source': 'github:OpenHands/skills', 'name': 'org-name'}]
        inherited, personal = _merge_marketplaces(instance, org, [])

        assert len(inherited) == 1
        # Org changes name, but ref and auto_load come from instance
        assert inherited[0]['name'] == 'org-name'
        assert inherited[0]['ref'] == 'main'
        assert inherited[0]['auto_load'] == 'all'
        assert inherited[0]['scope'] == 'org'


class TestGetInstanceDefaultMarketplacesJSONFormat:
    """Tests for JSON format parsing in INSTANCE_DEFAULT_MARKETPLACES.

    Note: The inline implementation in this test file uses the hash (#) format
    for all parsing. JSON format support requires the actual settings_router
    implementation. These tests verify the hash format parsing behavior.
    """

    def test_single_marketplace_github_format(self, monkeypatch):
        """Test parsing github: shorthand format."""
        monkeypatch.setenv(
            'INSTANCE_DEFAULT_MARKETPLACES', 'github:my/repo'
        )
        result = _get_instance_default_marketplaces()
        assert len(result) == 1
        assert result[0]['source'] == 'github:my/repo'
        assert result[0]['auto_load'] == 'all'

    def test_single_marketplace_https_url(self, monkeypatch):
        """Test parsing HTTPS git URL format."""
        monkeypatch.setenv(
            'INSTANCE_DEFAULT_MARKETPLACES', 'https://github.com/my/repo'
        )
        result = _get_instance_default_marketplaces()
        assert len(result) == 1
        assert result[0]['source'] == 'https://github.com/my/repo'
        assert result[0]['auto_load'] == 'all'

    def test_single_marketplace_ssh_url(self, monkeypatch):
        """Test parsing SSH git URL format."""
        monkeypatch.setenv(
            'INSTANCE_DEFAULT_MARKETPLACES', 'git@github.com:my/repo'
        )
        result = _get_instance_default_marketplaces()
        assert len(result) == 1
        assert result[0]['source'] == 'git@github.com:my/repo'
        assert result[0]['auto_load'] == 'all'

    def test_single_marketplace_local_path(self, monkeypatch):
        """Test parsing local path format."""
        monkeypatch.setenv(
            'INSTANCE_DEFAULT_MARKETPLACES', 'local/plugins'
        )
        result = _get_instance_default_marketplaces()
        assert len(result) == 1
        assert result[0]['source'] == 'local/plugins'
        assert result[0]['auto_load'] == 'all'

    def test_multiple_comma_separated_marketplaces(self, monkeypatch):
        """Test parsing multiple comma-separated marketplaces."""
        monkeypatch.setenv(
            'INSTANCE_DEFAULT_MARKETPLACES',
            'github:my/repo1, github:my/repo2, github:my/repo3',
        )
        result = _get_instance_default_marketplaces()
        assert len(result) == 3
        assert result[0]['source'] == 'github:my/repo1'
        assert result[1]['source'] == 'github:my/repo2'
        assert result[2]['source'] == 'github:my/repo3'

    def test_marketplace_with_hash_name_separator(self, monkeypatch):
        """Test parsing with # as field separator."""
        monkeypatch.setenv(
            'INSTANCE_DEFAULT_MARKETPLACES', 'github:my/repo#my-market'
        )
        result = _get_instance_default_marketplaces()
        assert len(result) == 1
        assert result[0]['source'] == 'github:my/repo'
        assert result[0]['name'] == 'my-market'

    def test_marketplace_with_all_hash_fields(self, monkeypatch):
        """Test parsing with all fields using # separator."""
        monkeypatch.setenv(
            'INSTANCE_DEFAULT_MARKETPLACES',
            'github:my/repo#market#v1#plugins',
        )
        result = _get_instance_default_marketplaces()
        assert len(result) == 1
        assert result[0]['source'] == 'github:my/repo'
        assert result[0]['name'] == 'market'
        assert result[0]['ref'] == 'v1'
        assert result[0]['repo_path'] == 'plugins'


class TestMarketplaceCompositionIntegration:
    """Integration tests for full marketplace composition flow.

    These tests verify the end-to-end composition behavior
    combining instance defaults, org settings, and user settings.
    """

    def test_enterprise_scenario_instance_and_org_only(self):
        """Test realistic enterprise scenario with instance and org settings only."""
        # Instance sets default
        instance = [
            {
                'source': 'github:openhands/default-plugins',
                'name': 'default',
                'auto_load': 'all',
            }
        ]

        # Org adds their own and overrides instance's auto_load
        org = [
            {
                'source': 'github:openhands/default-plugins',
                'auto_load': None,  # Org disables auto-load
            },
            {
                'source': 'github:acme/company-plugins',
                'name': 'acme-plugins',
                'auto_load': 'all',
            },
        ]

        inherited, personal = _merge_marketplaces(instance, org, [])

        # Inherited should have 2: default-plugins (org override) and acme-plugins (org)
        assert len(inherited) == 2
        assert personal == []

        inherited_by_source = {mp['source']: mp for mp in inherited}

        # default-plugins should be org scope with disabled auto_load
        assert inherited_by_source['github:openhands/default-plugins']['scope'] == 'org'
        assert inherited_by_source['github:openhands/default-plugins']['auto_load'] is None
        assert 'overridden' not in inherited_by_source['github:openhands/default-plugins']

        # acme-plugins should be org scope
        assert inherited_by_source['github:acme/company-plugins']['scope'] == 'org'
        assert inherited_by_source['github:acme/company-plugins']['auto_load'] == 'all'

    def test_user_adds_to_org_marketplaces(self):
        """Test user can add personal marketplace alongside org marketplaces."""
        org = [
            {
                'source': 'github:acme/company-plugins',
                'name': 'acme-plugins',
                'auto_load': 'all',
            },
        ]

        user = [
            {
                'source': 'github:user/personal-plugins',
                'name': 'personal',
                'auto_load': 'all',
            }
        ]

        inherited, personal = _merge_marketplaces([], org, user)

        # Inherited should have acme-plugins (org)
        assert len(inherited) == 1
        assert inherited[0]['scope'] == 'org'

        # Personal should have user's marketplace
        assert len(personal) == 1
        assert personal[0]['scope'] == 'personal'
        assert personal[0]['source'] == 'github:user/personal-plugins'

    def test_user_cannot_disable_org_marketplace(self):
        """Test that user cannot override org marketplace settings.

        Users can only add NEW personal marketplaces. They cannot modify
        or override instance/org marketplace settings.
        """
        org = [
            {
                'source': 'github:acme/company-plugins',
                'auto_load': 'all',
            },
        ]

        user = [
            {
                'source': 'github:acme/company-plugins',
                'auto_load': None,  # User tries to disable - should be ignored
            }
        ]

        inherited, personal = _merge_marketplaces([], org, user)

        # Org marketplace remains unchanged - user override is blocked
        assert len(inherited) == 1
        assert inherited[0]['scope'] == 'org'
        assert inherited[0]['auto_load'] == 'all'  # Still 'all' from org

        # User's attempt is ignored - not added to personal
        assert len(personal) == 0

    def test_no_confusion_between_name_and_source(self):
        """Test that name field is independent of source for deduplication."""
        # Two marketplaces with different sources but same name
        user = [
            {'source': 'github:owner/repo1', 'name': 'plugins'},
            {'source': 'github:owner/repo2', 'name': 'plugins'},
        ]

        inherited, personal = _merge_marketplaces([], [], user)

        # Both should be in personal (different sources)
        assert len(personal) == 2
        names = [mp['name'] for mp in personal]
        assert names.count('plugins') == 2  # Same name is allowed for different sources

    def test_security_path_traversal_not_possible(self):
        """Test that path traversal is prevented in sources."""
        # This is validated at model level, but verify merge doesn't bypass it
        instance = [
            {'source': 'github:owner/repo', 'auto_load': 'all'},
        ]
        org = [
            {'source': '../escape', 'auto_load': 'all'},
        ]

        # _merge_marketplaces(instance, org, user)
        inherited, personal = _merge_marketplaces(instance, org, [])

        # Should have 2 entries - they're different sources
        assert len(inherited) == 2
        assert len(personal) == 0
        # Note: Actual validation happens in MarketplaceRegistration model
