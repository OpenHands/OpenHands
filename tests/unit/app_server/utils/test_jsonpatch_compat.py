"""Tests for jsonpatch_compat utilities."""

from openhands.app_server.utils.jsonpatch_compat import (
    WHOLESALE_REPLACEMENT_KEYS,
    deep_merge,
    deep_merge_with_wholesale_keys,
)


class TestDeepMerge:
    """Tests for the base deep_merge function."""

    def test_basic_merge(self):
        base = {'a': 1, 'b': 2}
        updates = {'b': 3, 'c': 4}
        result = deep_merge(base, updates)
        assert result == {'a': 1, 'b': 3, 'c': 4}

    def test_nested_merge(self):
        base = {'outer': {'inner1': 1, 'inner2': 2}}
        updates = {'outer': {'inner2': 3, 'inner3': 4}}
        result = deep_merge(base, updates)
        assert result == {'outer': {'inner1': 1, 'inner2': 3, 'inner3': 4}}

    def test_none_removes_key(self):
        base = {'a': 1, 'b': 2}
        updates = {'b': None}
        result = deep_merge(base, updates)
        assert result == {'a': 1}

    def test_does_not_mutate_base(self):
        base = {'a': 1}
        updates = {'b': 2}
        deep_merge(base, updates)
        assert base == {'a': 1}


class TestDeepMergeWithWholesaleKeys:
    """Tests for deep_merge_with_wholesale_keys function."""

    def test_default_wholesale_keys(self):
        """Verify default wholesale keys are mcp_config and acp_env."""
        assert 'mcp_config' in WHOLESALE_REPLACEMENT_KEYS
        assert 'acp_env' in WHOLESALE_REPLACEMENT_KEYS

    def test_mcp_config_replaced_wholesale(self):
        """mcp_config should be replaced, not merged."""
        base = {
            'llm': {'model': 'gpt-4'},
            'mcp_config': {
                'mcpServers': {
                    'server1': {'url': 'https://s1.com'},
                    'server2': {'url': 'https://s2.com'},
                    'server3': {'url': 'https://s3.com'},
                }
            },
        }
        updates = {
            'mcp_config': {
                'mcpServers': {
                    'server1': {'url': 'https://s1.com'},
                    'server2': {'url': 'https://s2.com'},
                    # server3 deleted
                }
            }
        }

        result = deep_merge_with_wholesale_keys(base, updates)

        # server3 should NOT be resurrected
        assert len(result['mcp_config']['mcpServers']) == 2
        assert 'server3' not in result['mcp_config']['mcpServers']

    def test_acp_env_replaced_wholesale(self):
        """acp_env should be replaced, not merged."""
        base = {
            'acp_env': {
                'VAR1': 'value1',
                'VAR2': 'value2',
                'VAR3': 'value3',
            }
        }
        updates = {
            'acp_env': {
                'VAR1': 'value1',
                'VAR2': 'value2',
                # VAR3 deleted
            }
        }

        result = deep_merge_with_wholesale_keys(base, updates)

        assert len(result['acp_env']) == 2
        assert 'VAR3' not in result['acp_env']

    def test_other_keys_still_deep_merged(self):
        """Non-wholesale keys should still be deep merged."""
        base = {
            'llm': {'model': 'gpt-4', 'temperature': 0.7},
            'mcp_config': {'mcpServers': {'old': {}}},
        }
        updates = {
            'llm': {'model': 'gpt-5'},  # should merge
            'mcp_config': {'mcpServers': {'new': {}}},  # should replace
        }

        result = deep_merge_with_wholesale_keys(base, updates)

        # llm should be deep merged (temperature preserved)
        assert result['llm']['model'] == 'gpt-5'
        assert result['llm']['temperature'] == 0.7

        # mcp_config should be replaced (old server gone)
        assert 'old' not in result['mcp_config']['mcpServers']
        assert 'new' in result['mcp_config']['mcpServers']

    def test_wholesale_key_not_in_updates(self):
        """If wholesale key not in updates, it should be preserved from base."""
        base = {
            'llm': {'model': 'gpt-4'},
            'mcp_config': {'mcpServers': {'existing': {'url': 'https://existing.com'}}},
        }
        updates = {
            'llm': {'model': 'gpt-5'}
            # mcp_config NOT in updates
        }

        result = deep_merge_with_wholesale_keys(base, updates)

        # mcp_config should be preserved (deep merged, not cleared)
        assert 'mcp_config' in result
        assert 'existing' in result['mcp_config']['mcpServers']

    def test_empty_wholesale_value(self):
        """Empty dict for wholesale key should clear it."""
        base = {
            'mcp_config': {
                'mcpServers': {
                    'server1': {'url': 'https://s1.com'},
                    'server2': {'url': 'https://s2.com'},
                }
            }
        }
        updates = {
            'mcp_config': {
                'mcpServers': {}  # delete all servers
            }
        }

        result = deep_merge_with_wholesale_keys(base, updates)

        assert result['mcp_config']['mcpServers'] == {}

    def test_custom_wholesale_keys(self):
        """Should support custom wholesale keys via parameter."""
        base = {'custom_dict': {'a': 1, 'b': 2, 'c': 3}}
        updates = {
            'custom_dict': {'a': 1, 'b': 2}  # c deleted
        }

        # Without custom keys, c would be preserved (deep merge)
        result_default = deep_merge_with_wholesale_keys(base, updates)
        assert 'c' in result_default['custom_dict']

        # With custom keys, c should be deleted (wholesale replace)
        result_custom = deep_merge_with_wholesale_keys(
            base, updates, wholesale_keys=frozenset({'custom_dict'})
        )
        assert 'c' not in result_custom['custom_dict']

    def test_does_not_mutate_inputs(self):
        """Should not mutate base or updates."""
        base = {'mcp_config': {'mcpServers': {'s1': {}}}}
        updates = {'mcp_config': {'mcpServers': {'s2': {}}}}

        base_copy = {'mcp_config': {'mcpServers': {'s1': {}}}}
        updates_copy = {'mcp_config': {'mcpServers': {'s2': {}}}}

        deep_merge_with_wholesale_keys(base, updates)

        assert base == base_copy
        assert updates == updates_copy
