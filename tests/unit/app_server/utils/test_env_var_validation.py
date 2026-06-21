"""Tests for environment variable name validation utility."""

import pytest

from openhands.app_server.utils.env_var_validation import (
    is_valid_env_var_name,
    validate_env_var_name,
)


class TestIsValidEnvVarName:
    @pytest.mark.parametrize(
        'name',
        [
            'MY_VAR',
            'my_var',
            'MyVar',
            '_PRIVATE',
            '_',
            '__',
            'A',
            'a',
            'VAR123',
            '_123',
            'API_KEY',
            'DATABASE_URL',
            'GITHUB_TOKEN',
        ],
    )
    def test_valid_names(self, name: str):
        assert is_valid_env_var_name(name) is True

    @pytest.mark.parametrize(
        'name',
        [
            'MY-VAR',
            'MY VAR',
            'MY.VAR',
            '123VAR',
            '1',
            '-VAR',
            'MY@VAR',
            'MY$VAR',
            'MY#VAR',
            'MY!VAR',
            'MY%VAR',
            'MY^VAR',
            'MY&VAR',
            'MY*VAR',
            'MY(VAR',
            'MY)VAR',
            'MY+VAR',
            'MY=VAR',
            'MY[VAR',
            'MY]VAR',
            'MY{VAR',
            'MY}VAR',
            'MY|VAR',
            'MY\\VAR',
            'MY/VAR',
            'MY?VAR',
            'MY<VAR',
            'MY>VAR',
            'MY,VAR',
            'MY:VAR',
            'MY;VAR',
            "MY'VAR",
            'MY"VAR',
            'MY`VAR',
            'MY~VAR',
            'MY_VAR\n',
        ],
    )
    def test_invalid_names_special_chars(self, name: str):
        assert is_valid_env_var_name(name) is False

    def test_empty_string(self):
        assert is_valid_env_var_name('') is False


class TestValidateEnvVarName:
    def test_valid_name_passes(self):
        validate_env_var_name('MY_VAR')  # should not raise

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match='Invalid'):
            validate_env_var_name('MY-VAR')

    def test_custom_field_name_in_error(self):
        with pytest.raises(ValueError, match='secret name'):
            validate_env_var_name('MY-VAR', field_name='secret name')
