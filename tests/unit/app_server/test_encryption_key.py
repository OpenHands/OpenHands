"""Tests for encryption key utilities.

This module tests the get_default_encryption_keys function which handles:
- Loading keys from JWT_SECRET environment variable
- Loading keys from .keys JSON file
- Loading keys from legacy .jwt_secret file
- Generating new keys when none exist
"""

import datetime
import hashlib
import os
from unittest.mock import patch

import base62
import pytest
from pydantic import SecretStr, TypeAdapter

from openhands.app_server.utils.encryption_key import (
    EncryptionKey,
    get_default_encryption_keys,
)


class TestGetDefaultEncryptionKeys:
    """Test cases for get_default_encryption_keys function."""

    @pytest.fixture
    def temp_workspace(self, tmp_path):
        """Create a temporary workspace directory."""
        return tmp_path

    @pytest.fixture
    def clear_jwt_secret_env(self):
        """Ensure JWT_SECRET env var is not set during tests."""
        original = os.environ.pop('JWT_SECRET', None)
        yield
        if original is not None:
            os.environ['JWT_SECRET'] = original

    def test_jwt_secret_env_var_takes_priority(self, temp_workspace):
        """When JWT_SECRET env var is set, it takes priority over files."""
        jwt_secret = 'my-super-secret-key-from-env'
        expected_key_id = base62.encodebytes(
            hashlib.sha256(jwt_secret.encode()).digest()
        )

        with patch.dict(os.environ, {'JWT_SECRET': jwt_secret}):
            keys = get_default_encryption_keys(temp_workspace)

        assert len(keys) == 1
        assert keys[0].id == expected_key_id
        assert keys[0].key.get_secret_value() == jwt_secret
        assert keys[0].active is True
        assert keys[0].notes == 'jwt secret master key'

    def test_jwt_secret_env_var_ignores_files(self, temp_workspace):
        """When JWT_SECRET env var is set, .keys and .jwt_secret files are ignored."""
        # Create both files
        keys_file = temp_workspace / '.keys'
        jwt_secret_file = temp_workspace / '.jwt_secret'

        file_key = EncryptionKey(
            id='file-key-id',
            key=SecretStr('file-secret'),
            active=True,
            notes='from file',
        )
        type_adapter = TypeAdapter(list[EncryptionKey])
        keys_file.write_bytes(
            type_adapter.dump_json([file_key], context={'expose_secrets': True})
        )
        jwt_secret_file.write_text('legacy-jwt-secret')

        env_secret = 'env-var-secret'
        with patch.dict(os.environ, {'JWT_SECRET': env_secret}):
            keys = get_default_encryption_keys(temp_workspace)

        # Should only have the env var key, not the file keys
        assert len(keys) == 1
        assert keys[0].key.get_secret_value() == env_secret

    def test_loads_keys_from_keys_file(self, temp_workspace, clear_jwt_secret_env):
        """When .keys file exists, loads keys from it."""
        keys_file = temp_workspace / '.keys'

        stored_keys = [
            EncryptionKey(
                id='key-1',
                key=SecretStr('secret-1'),
                active=True,
                notes='first key',
                created_at=datetime.datetime(2023, 1, 1, tzinfo=datetime.UTC),
            ),
            EncryptionKey(
                id='key-2',
                key=SecretStr('secret-2'),
                active=False,
                notes='rotated key',
                created_at=datetime.datetime(2023, 6, 1, tzinfo=datetime.UTC),
            ),
        ]
        type_adapter = TypeAdapter(list[EncryptionKey])
        keys_file.write_bytes(
            type_adapter.dump_json(stored_keys, context={'expose_secrets': True})
        )

        keys = get_default_encryption_keys(temp_workspace)

        assert len(keys) == 2
        assert keys[0].id == 'key-1'
        assert keys[0].key.get_secret_value() == 'secret-1'
        assert keys[0].active is True
        assert keys[1].id == 'key-2'
        assert keys[1].key.get_secret_value() == 'secret-2'
        assert keys[1].active is False

    def test_loads_keys_from_jwt_secret_file(
        self, temp_workspace, clear_jwt_secret_env
    ):
        """When only .jwt_secret file exists, loads key from it."""
        jwt_secret_file = temp_workspace / '.jwt_secret'
        jwt_secret = 'legacy-jwt-secret-value'
        jwt_secret_file.write_text(jwt_secret)

        expected_key_id = base62.encodebytes(
            hashlib.sha256(jwt_secret.encode()).digest()
        )

        keys = get_default_encryption_keys(temp_workspace)

        assert len(keys) == 1
        assert keys[0].id == expected_key_id
        assert keys[0].key.get_secret_value() == jwt_secret
        assert keys[0].active is True
        assert keys[0].notes == 'jwt secret master key'

    def test_jwt_secret_file_strips_whitespace(
        self, temp_workspace, clear_jwt_secret_env
    ):
        """The .jwt_secret file content should be stripped of whitespace."""
        jwt_secret_file = temp_workspace / '.jwt_secret'
        jwt_secret = 'secret-with-whitespace'
        jwt_secret_file.write_text(f'  {jwt_secret}  \n')

        keys = get_default_encryption_keys(temp_workspace)

        assert keys[0].key.get_secret_value() == jwt_secret

    def test_combines_keys_and_jwt_secret_files(
        self, temp_workspace, clear_jwt_secret_env
    ):
        """When both .keys and .jwt_secret exist, combines keys from both."""
        keys_file = temp_workspace / '.keys'
        jwt_secret_file = temp_workspace / '.jwt_secret'

        stored_key = EncryptionKey(
            id='keys-file-key',
            key=SecretStr('keys-file-secret'),
            active=True,
            notes='from .keys file',
        )
        type_adapter = TypeAdapter(list[EncryptionKey])
        keys_file.write_bytes(
            type_adapter.dump_json([stored_key], context={'expose_secrets': True})
        )

        jwt_secret = 'jwt-secret-file-value'
        jwt_secret_file.write_text(jwt_secret)

        keys = get_default_encryption_keys(temp_workspace)

        # Should have both keys
        assert len(keys) == 2
        assert keys[0].id == 'keys-file-key'
        assert keys[0].key.get_secret_value() == 'keys-file-secret'
        # Second key is from .jwt_secret
        assert keys[1].key.get_secret_value() == jwt_secret

    def test_generates_new_key_when_none_exist(
        self, temp_workspace, clear_jwt_secret_env
    ):
        """When no keys exist, generates a new one and persists it."""
        keys_file = temp_workspace / '.keys'
        assert not keys_file.exists()

        keys = get_default_encryption_keys(temp_workspace)

        # Should generate one key
        assert len(keys) == 1
        assert keys[0].active is True
        assert keys[0].notes == 'generated master key'
        # Key should be non-empty
        assert len(keys[0].key.get_secret_value()) > 0

        # Should persist to .keys file
        assert keys_file.exists()

    def test_generated_key_is_persisted_correctly(
        self, temp_workspace, clear_jwt_secret_env
    ):
        """Generated keys should be readable on subsequent calls."""
        # First call generates key
        keys1 = get_default_encryption_keys(temp_workspace)

        # Second call should load the same key
        keys2 = get_default_encryption_keys(temp_workspace)

        assert len(keys1) == 1
        assert len(keys2) == 1
        assert keys1[0].id == keys2[0].id
        assert keys1[0].key.get_secret_value() == keys2[0].key.get_secret_value()

    def test_jwt_secret_file_created_at_uses_file_mtime(
        self, temp_workspace, clear_jwt_secret_env
    ):
        """The created_at for .jwt_secret key should use the file's mtime."""
        jwt_secret_file = temp_workspace / '.jwt_secret'
        jwt_secret_file.write_text('test-secret')

        # Get the file's modification time
        file_mtime = jwt_secret_file.stat().st_mtime
        expected_created_at = datetime.datetime.fromtimestamp(
            file_mtime, tz=datetime.UTC
        )

        keys = get_default_encryption_keys(temp_workspace)

        assert keys[0].created_at == expected_created_at

    def test_deterministic_key_id_from_jwt_secret_env(self, temp_workspace):
        """Key ID derived from JWT_SECRET should be deterministic."""
        jwt_secret = 'deterministic-secret'
        expected_key_id = base62.encodebytes(
            hashlib.sha256(jwt_secret.encode()).digest()
        )

        with patch.dict(os.environ, {'JWT_SECRET': jwt_secret}):
            keys1 = get_default_encryption_keys(temp_workspace)
            keys2 = get_default_encryption_keys(temp_workspace)

        assert keys1[0].id == keys2[0].id == expected_key_id

    def test_deterministic_key_id_from_jwt_secret_file(
        self, temp_workspace, clear_jwt_secret_env
    ):
        """Key ID derived from .jwt_secret file should be deterministic."""
        jwt_secret = 'file-based-secret'
        jwt_secret_file = temp_workspace / '.jwt_secret'
        jwt_secret_file.write_text(jwt_secret)

        expected_key_id = base62.encodebytes(
            hashlib.sha256(jwt_secret.encode()).digest()
        )

        keys = get_default_encryption_keys(temp_workspace)

        assert keys[0].id == expected_key_id


class TestEncryptionKey:
    """Test cases for EncryptionKey model."""

    def test_default_id_is_generated(self):
        """EncryptionKey generates a random ID by default."""
        key1 = EncryptionKey(key=SecretStr('secret1'))
        key2 = EncryptionKey(key=SecretStr('secret2'))

        # IDs should be non-empty and different
        assert key1.id
        assert key2.id
        assert key1.id != key2.id

    def test_default_active_is_true(self):
        """EncryptionKey defaults to active=True."""
        key = EncryptionKey(key=SecretStr('secret'))
        assert key.active is True

    def test_serialize_key_masks_by_default(self):
        """Key serialization masks the secret by default."""
        key = EncryptionKey(key=SecretStr('super-secret'))
        serialized = key.model_dump()

        # The key should be masked
        assert serialized['key'] == '**********'

    def test_serialize_key_exposes_with_context(self):
        """Key serialization exposes secret when context allows."""
        key = EncryptionKey(key=SecretStr('super-secret'))
        serialized = key.model_dump(context={'expose_secrets': True})

        assert serialized['key'] == 'super-secret'

    def test_created_at_defaults_to_now(self):
        """EncryptionKey created_at defaults to current time."""
        before = datetime.datetime.now(datetime.UTC)
        key = EncryptionKey(key=SecretStr('secret'))
        after = datetime.datetime.now(datetime.UTC)

        assert before <= key.created_at <= after
