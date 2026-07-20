"""Tests for marketplace composition.

Covers :mod:`openhands.app_server.settings.marketplace_composition`: instance
env parsing, name-keyed additive composition across scopes, duplicate detection,
and the loading feature flag.
"""

from __future__ import annotations

import pytest

from openhands.app_server.settings.marketplace_composition import (
    compose_marketplaces,
    duplicate_marketplace_names,
    get_instance_default_marketplaces,
    load_composed_marketplaces,
    marketplace_plugin_loading_enabled,
)
from openhands.app_server.settings.settings_models import MarketplaceScope

ENV = 'INSTANCE_DEFAULT_MARKETPLACES'
FLAG = 'ENABLE_MARKETPLACE_PLUGIN_LOADING'


class TestGetInstanceDefaultMarketplaces:
    """Parsing of the INSTANCE_DEFAULT_MARKETPLACES environment variable."""

    def test_unset_returns_empty(self, monkeypatch):
        # Arrange
        monkeypatch.delenv(ENV, raising=False)
        # Act / Assert
        assert get_instance_default_marketplaces() == []

    def test_hash_format_parses_all_fields_and_auto_loads(self, monkeypatch):
        # Arrange
        monkeypatch.setenv(ENV, 'github:owner/repo#team#main#marketplaces/internal')
        # Act
        result = get_instance_default_marketplaces()
        # Assert
        assert result == [
            {
                'name': 'team',
                'source': 'github:owner/repo',
                'ref': 'main',
                'repo_path': 'marketplaces/internal',
                'auto_load': True,
                'scope': None,
            }
        ]

    def test_name_derived_from_source_when_omitted(self, monkeypatch):
        # Arrange
        monkeypatch.setenv(ENV, 'github:owner/my-repo')
        # Act
        result = get_instance_default_marketplaces()
        # Assert
        assert result[0]['name'] == 'my-repo'

    def test_multiple_comma_separated_entries(self, monkeypatch):
        # Arrange
        monkeypatch.setenv(ENV, 'github:a/one#one, github:b/two#two')
        # Act
        result = get_instance_default_marketplaces()
        # Assert
        assert [m['name'] for m in result] == ['one', 'two']

    def test_json_object_with_commas_is_parsed_whole(self, monkeypatch):
        # Arrange - a JSON object contains commas; it must not be comma-split.
        monkeypatch.setenv(
            ENV, '{"source": "github:acme/plugins", "name": "team", "ref": "main"}'
        )
        # Act
        result = get_instance_default_marketplaces()
        # Assert
        assert len(result) == 1
        assert result[0]['name'] == 'team'
        assert result[0]['ref'] == 'main'

    def test_json_list_defaults_auto_load_true(self, monkeypatch):
        # Arrange
        monkeypatch.setenv(ENV, '[{"source": "github:acme/plugins", "name": "team"}]')
        # Act
        result = get_instance_default_marketplaces()
        # Assert
        assert result[0]['auto_load'] is True

    def test_invalid_config_degrades_to_empty(self, monkeypatch):
        # Arrange - malformed JSON (non-dict list) must not raise.
        monkeypatch.setenv(ENV, '["not-an-object"]')
        # Act / Assert
        assert get_instance_default_marketplaces() == []


class TestComposeMarketplaces:
    """Name-keyed additive composition across instance/org/user scopes."""

    def test_instance_entries_are_inherited_with_instance_scope(self):
        # Arrange
        instance = [{'name': 'pub', 'source': 'github:o/pub', 'auto_load': True}]
        # Act
        composed = compose_marketplaces(instance, [], [])
        # Assert
        assert [(m.name, m.scope) for m in composed.inherited] == [
            ('pub', MarketplaceScope.INSTANCE)
        ]
        assert composed.personal == []

    def test_instance_and_org_are_additive(self):
        # Arrange
        instance = [{'name': 'pub', 'source': 'github:o/pub'}]
        org = [{'name': 'team', 'source': 'github:o/team'}]
        # Act
        composed = compose_marketplaces(instance, org, [])
        # Assert
        assert [m.name for m in composed.inherited] == ['pub', 'team']

    def test_org_overrides_instance_by_name(self):
        # Arrange - org re-declares 'pub' to flip auto_load (AC #7).
        instance = [{'name': 'pub', 'source': 'github:o/pub', 'auto_load': True}]
        org = [{'name': 'pub', 'source': 'github:o/pub', 'auto_load': False}]
        # Act
        composed = compose_marketplaces(instance, org, [])
        # Assert
        assert len(composed.inherited) == 1
        assert composed.inherited[0].scope == MarketplaceScope.ORG
        assert composed.inherited[0].auto_load is False

    def test_user_adds_new_personal_marketplace(self):
        # Arrange
        user = [{'name': 'mine', 'source': 'github:o/mine'}]
        # Act
        composed = compose_marketplaces([], [], user)
        # Assert
        assert [(m.name, m.scope) for m in composed.personal] == [
            ('mine', MarketplaceScope.PERSONAL)
        ]

    def test_user_cannot_shadow_inherited_name(self):
        # Arrange - user 'team' collides with an org marketplace of the same name.
        org = [{'name': 'team', 'source': 'github:o/team'}]
        user = [{'name': 'team', 'source': 'github:o/fork'}]
        # Act
        composed = compose_marketplaces([], org, user)
        # Assert - org wins, user entry dropped.
        assert composed.inherited[0].source == 'github:o/team'
        assert composed.personal == []

    def test_invalid_entry_is_skipped(self):
        # Arrange - instance entry missing required name.
        composed = compose_marketplaces(
            [{'source': 'github:o/x'}], [], [{'name': 'ok', 'source': 'github:o/ok'}]
        )
        # Assert
        assert composed.inherited == []
        assert [m.name for m in composed.personal] == ['ok']

    def test_duplicate_names_deduped_defensively(self):
        # Arrange - two personal entries share a name (bad stored data).
        user = [
            {'name': 'dup', 'source': 'github:o/a'},
            {'name': 'dup', 'source': 'github:o/b'},
        ]
        # Act - must not raise; last one wins.
        composed = compose_marketplaces([], [], user)
        # Assert
        assert len(composed.personal) == 1
        assert composed.personal[0].source == 'github:o/b'

    def test_all_combines_inherited_and_personal(self):
        # Arrange
        instance = [{'name': 'pub', 'source': 'github:o/pub'}]
        user = [{'name': 'mine', 'source': 'github:o/mine'}]
        # Act
        composed = compose_marketplaces(instance, [], user)
        # Assert
        assert [m.name for m in composed.all] == ['pub', 'mine']


class TestDuplicateMarketplaceNames:
    """Detection of duplicate / reserved marketplace names for write validation."""

    def test_duplicates_within_list(self):
        # Arrange
        marketplaces = [{'name': 'a', 'source': 'x'}, {'name': 'a', 'source': 'y'}]
        # Act / Assert
        assert duplicate_marketplace_names(marketplaces) == ['a']

    def test_reserved_name_conflict(self):
        # Arrange
        marketplaces = [{'name': 'team', 'source': 'x'}]
        # Act / Assert
        assert duplicate_marketplace_names(marketplaces, reserved_names=['team']) == [
            'team'
        ]

    def test_no_conflicts(self):
        # Arrange
        marketplaces = [{'name': 'a', 'source': 'x'}, {'name': 'b', 'source': 'y'}]
        # Act / Assert
        assert duplicate_marketplace_names(marketplaces, reserved_names=['c']) == []


class TestMarketplaceLoadingFlag:
    """The ENABLE_MARKETPLACE_PLUGIN_LOADING gate."""

    def test_enabled_by_default(self, monkeypatch):
        # Arrange
        monkeypatch.delenv(FLAG, raising=False)
        # Act / Assert
        assert marketplace_plugin_loading_enabled() is True

    def test_disabled_when_falsy(self, monkeypatch):
        # Arrange
        monkeypatch.setenv(FLAG, 'false')
        # Act / Assert
        assert marketplace_plugin_loading_enabled() is False

    def test_enabled_when_truthy(self, monkeypatch):
        # Arrange
        monkeypatch.setenv(FLAG, 'true')
        # Act / Assert
        assert marketplace_plugin_loading_enabled() is True


class _FakeStore:
    """Minimal settings store exposing get_org_marketplaces."""

    def __init__(self, org_marketplaces):
        self._org = org_marketplaces

    async def get_org_marketplaces(self, user_id):
        return self._org


class TestLoadComposedMarketplaces:
    """The async gatherer that pulls instance + org + user and composes them."""

    @pytest.mark.asyncio
    async def test_gathers_all_three_scopes(self, monkeypatch):
        # Arrange
        monkeypatch.setenv(ENV, 'github:o/pub#pub')
        store = _FakeStore([{'name': 'team', 'source': 'github:o/team'}])
        user = [{'name': 'mine', 'source': 'github:o/mine'}]
        # Act
        composed = await load_composed_marketplaces('user-1', user, store)
        # Assert
        assert {m.name for m in composed.inherited} == {'pub', 'team'}
        assert [m.name for m in composed.personal] == ['mine']

    @pytest.mark.asyncio
    async def test_org_lookup_failure_degrades(self, monkeypatch):
        # Arrange - org lookup raising must not break composition.
        monkeypatch.delenv(ENV, raising=False)

        class _BrokenStore:
            async def get_org_marketplaces(self, user_id):
                raise RuntimeError('db down')

        # Act
        composed = await load_composed_marketplaces(
            'user-1', [{'name': 'mine', 'source': 'github:o/mine'}], _BrokenStore()
        )
        # Assert
        assert composed.inherited == []
        assert [m.name for m in composed.personal] == ['mine']
