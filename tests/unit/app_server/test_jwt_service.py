"""Tests for JwtService.

This module tests the JWT service functionality including:
- JWS token creation and verification (sign/verify round trip)
- JWE token creation and decryption (encrypt/decrypt round trip)
- Key management and rotation
- Error handling and edge cases
"""

import json
from datetime import datetime, timedelta
from unittest.mock import patch

import jwt
import pytest
from jose import jwe
from pydantic import SecretStr

from openhands.app_server.services.jwt_service import JwtService
from openhands.app_server.utils.encryption_key import EncryptionKey


class TestJwtService:
    """Test cases for JwtService."""

    @pytest.fixture
    def sample_keys(self):
        """Create sample encryption keys for testing."""
        return [
            EncryptionKey(
                id='key1',
                key=SecretStr('test_secret_key_1'),
                active=True,
                notes='Test key 1',
                created_at=datetime(2023, 1, 1, tzinfo=None),
            ),
            EncryptionKey(
                id='key2',
                key=SecretStr('test_secret_key_2'),
                active=True,
                notes='Test key 2',
                created_at=datetime(2023, 1, 2, tzinfo=None),
            ),
            EncryptionKey(
                id='key3',
                key=SecretStr('test_secret_key_3'),
                active=False,
                notes='Inactive test key',
                created_at=datetime(2023, 1, 3, tzinfo=None),
            ),
        ]

    @pytest.fixture
    def jwt_service(self, sample_keys):
        """Create a JwtService instance with sample keys."""
        return JwtService(sample_keys)

    def test_initialization_with_valid_keys(self, sample_keys):
        """Test JwtService initialization with valid keys."""
        service = JwtService(sample_keys)

        # Should use the newest active key as default
        assert service.default_key_id == 'key2'

    def test_initialization_no_active_keys(self):
        """Test JwtService initialization fails with no active keys."""
        inactive_keys = [
            EncryptionKey(
                id='key1',
                key=SecretStr('test_key'),
                active=False,
                notes='Inactive key',
            )
        ]

        with pytest.raises(ValueError, match='At least one active key is required'):
            JwtService(inactive_keys)

    def test_initialization_empty_keys(self):
        """Test JwtService initialization fails with empty key list."""
        with pytest.raises(ValueError, match='At least one active key is required'):
            JwtService([])

    def test_jws_token_round_trip_default_key(self, jwt_service):
        """Test JWS token creation and verification round trip with default key."""
        payload = {'user_id': '123', 'role': 'admin', 'custom_data': {'foo': 'bar'}}

        # Create token
        token = jwt_service.create_jws_token(payload)

        # Verify token
        decoded_payload = jwt_service.verify_jws_token(token)

        # Check that original payload is preserved
        assert decoded_payload['user_id'] == payload['user_id']
        assert decoded_payload['role'] == payload['role']
        assert decoded_payload['custom_data'] == payload['custom_data']

        # Check that standard JWT claims are added
        assert 'iat' in decoded_payload
        assert 'exp' in decoded_payload
        # JWT library converts datetime to Unix timestamps
        assert isinstance(decoded_payload['iat'], int)
        assert isinstance(decoded_payload['exp'], int)

    def test_jws_token_round_trip_specific_key(self, jwt_service):
        """Test JWS token creation and verification with specific key."""
        payload = {'user_id': '456', 'permissions': ['read', 'write']}

        # Create token with specific key
        token = jwt_service.create_jws_token(payload, key_id='key1')

        # Verify token (should auto-detect key from header)
        decoded_payload = jwt_service.verify_jws_token(token)

        # Check payload
        assert decoded_payload['user_id'] == payload['user_id']
        assert decoded_payload['permissions'] == payload['permissions']

    def test_jws_token_round_trip_with_expiration(self, jwt_service):
        """Test JWS token creation and verification with custom expiration."""
        payload = {'user_id': '789'}
        expires_in = timedelta(minutes=30)

        # Create token with custom expiration
        token = jwt_service.create_jws_token(payload, expires_in=expires_in)

        # Verify token
        decoded_payload = jwt_service.verify_jws_token(token)

        # Check that expiration is set correctly (within reasonable tolerance)
        exp_time = decoded_payload['exp']
        iat_time = decoded_payload['iat']
        actual_duration = exp_time - iat_time  # Both are Unix timestamps (integers)

        # Allow for small timing differences
        assert abs(actual_duration - expires_in.total_seconds()) < 1

    def test_jws_token_invalid_key_id(self, jwt_service):
        """Test JWS token creation fails with invalid key ID."""
        payload = {'user_id': '123'}

        with pytest.raises(ValueError, match="Key ID 'invalid_key' not found"):
            jwt_service.create_jws_token(payload, key_id='invalid_key')

    def test_jws_token_verification_invalid_key_id(self, jwt_service):
        """Test JWS token verification fails with invalid key ID."""
        payload = {'user_id': '123'}
        token = jwt_service.create_jws_token(payload)

        with pytest.raises(ValueError, match="Key ID 'invalid_key' not found"):
            jwt_service.verify_jws_token(token, key_id='invalid_key')

    def test_jws_token_verification_malformed_token(self, jwt_service):
        """Test JWS token verification fails with malformed token."""
        with pytest.raises(ValueError, match='Invalid JWT token format'):
            jwt_service.verify_jws_token('invalid.token')

    def test_jws_token_verification_no_kid_header(self, jwt_service):
        """Test JWS token verification fails when token has no kid header."""
        # Create a token without kid header using PyJWT directly
        payload = {'user_id': '123'}
        token = jwt.encode(payload, 'some_secret', algorithm='HS256')

        with pytest.raises(
            ValueError, match="Token does not contain 'kid' header with key ID"
        ):
            jwt_service.verify_jws_token(token)

    def test_jws_token_verification_wrong_signature(self, jwt_service):
        """Test JWS token verification fails with wrong signature."""
        payload = {'user_id': '123'}

        # Create token with one key
        token = jwt_service.create_jws_token(payload, key_id='key1')

        # Try to verify with different key
        with pytest.raises(jwt.InvalidTokenError, match='Token verification failed'):
            jwt_service.verify_jws_token(token, key_id='key2')

    def test_jwe_token_round_trip_default_key(self, jwt_service):
        """Test JWE token creation and decryption round trip with default key."""
        payload = {
            'user_id': '123',
            'sensitive_data': 'secret_info',
            'nested': {'key': 'value'},
        }

        # Create encrypted token
        token = jwt_service.create_jwe_token(payload)

        # Decrypt token
        decrypted_payload = jwt_service.decrypt_jwe_token(token)

        # Check that original payload is preserved
        assert decrypted_payload['user_id'] == payload['user_id']
        assert decrypted_payload['sensitive_data'] == payload['sensitive_data']
        assert decrypted_payload['nested'] == payload['nested']

        # Check that standard JWT claims are added
        assert 'iat' in decrypted_payload
        assert 'exp' in decrypted_payload
        assert isinstance(decrypted_payload['iat'], int)  # JWE uses timestamp integers
        assert isinstance(decrypted_payload['exp'], int)

    def test_jwe_token_round_trip_specific_key(self, jwt_service):
        """Test JWE token creation and decryption with specific key."""
        payload = {'confidential': 'data', 'array': [1, 2, 3]}

        # Create encrypted token with specific key
        token = jwt_service.create_jwe_token(payload, key_id='key1')

        # Decrypt token (should auto-detect key from header)
        decrypted_payload = jwt_service.decrypt_jwe_token(token)

        # Check payload
        assert decrypted_payload['confidential'] == payload['confidential']
        assert decrypted_payload['array'] == payload['array']

    def test_jwe_token_round_trip_with_expiration(self, jwt_service):
        """Test JWE token creation and decryption with custom expiration."""
        payload = {'user_id': '789'}
        expires_in = timedelta(hours=2)

        # Create encrypted token with custom expiration
        token = jwt_service.create_jwe_token(payload, expires_in=expires_in)

        # Decrypt token
        decrypted_payload = jwt_service.decrypt_jwe_token(token)

        # Check that expiration is set correctly (within reasonable tolerance)
        exp_time = decrypted_payload['exp']
        iat_time = decrypted_payload['iat']
        actual_duration = exp_time - iat_time

        # Allow for small timing differences
        assert abs(actual_duration - expires_in.total_seconds()) < 1

    def test_jwe_token_invalid_key_id(self, jwt_service):
        """Test JWE token creation fails with invalid key ID."""
        payload = {'user_id': '123'}

        with pytest.raises(ValueError, match="Key ID 'invalid_key' not found"):
            jwt_service.create_jwe_token(payload, key_id='invalid_key')

    def test_jwe_token_decryption_invalid_key_id(self, jwt_service):
        """Test JWE token decryption fails with invalid key ID."""
        payload = {'user_id': '123'}
        token = jwt_service.create_jwe_token(payload)

        with pytest.raises(ValueError, match="Key ID 'invalid_key' not found"):
            jwt_service.decrypt_jwe_token(token, key_id='invalid_key')

    def test_jwe_token_decryption_malformed_token(self, jwt_service):
        """Test JWE token decryption fails with malformed token."""
        with pytest.raises(ValueError, match='Invalid JWE token format'):
            jwt_service.decrypt_jwe_token('invalid.token')

    def test_jwe_token_decryption_no_kid_header(self, jwt_service):
        """Test JWE token decryption fails when token has no kid header."""
        # Create a JWE token without kid header using python-jose directly
        payload = {'user_id': '123'}
        # Create a proper 32-byte key for A256GCM
        key = b'12345678901234567890123456789012'  # Exactly 32 bytes

        token = jwe.encrypt(
            json.dumps(payload), key, algorithm='dir', encryption='A256GCM'
        )

        with pytest.raises(ValueError, match='Invalid JWE token format'):
            jwt_service.decrypt_jwe_token(token)

    def test_jwe_token_decryption_wrong_key(self, jwt_service):
        """Test JWE token decryption fails with wrong key."""
        payload = {'user_id': '123'}

        # Create token with one key
        token = jwt_service.create_jwe_token(payload, key_id='key1')

        # Try to decrypt with different key
        with pytest.raises(Exception, match='Token decryption failed'):
            jwt_service.decrypt_jwe_token(token, key_id='key2')

    def test_jws_and_jwe_tokens_are_different(self, jwt_service):
        """Test that JWS and JWE tokens for same payload are different."""
        payload = {'user_id': '123', 'data': 'test'}

        jws_token = jwt_service.create_jws_token(payload)
        jwe_token = jwt_service.create_jwe_token(payload)

        # Tokens should be different
        assert jws_token != jwe_token

        # JWS token should be readable without decryption (just verification)
        jws_decoded = jwt_service.verify_jws_token(jws_token)
        assert jws_decoded['user_id'] == payload['user_id']

        # JWE token should require decryption
        jwe_decrypted = jwt_service.decrypt_jwe_token(jwe_token)
        assert jwe_decrypted['user_id'] == payload['user_id']

    def test_key_rotation_scenario(self, jwt_service):
        """Test key rotation scenario where tokens created with different keys can be verified."""
        payload = {'user_id': '123'}

        # Create tokens with different keys
        token_key1 = jwt_service.create_jws_token(payload, key_id='key1')
        token_key2 = jwt_service.create_jws_token(payload, key_id='key2')

        # Both tokens should be verifiable
        decoded1 = jwt_service.verify_jws_token(token_key1)
        decoded2 = jwt_service.verify_jws_token(token_key2)

        assert decoded1['user_id'] == payload['user_id']
        assert decoded2['user_id'] == payload['user_id']

    def test_complex_payload_structures(self, jwt_service):
        """Test JWS and JWE with complex payload structures."""
        complex_payload = {
            'user_id': 'user123',
            'metadata': {
                'permissions': ['read', 'write', 'admin'],
                'settings': {
                    'theme': 'dark',
                    'notifications': True,
                    'nested_array': [
                        {'id': 1, 'name': 'item1'},
                        {'id': 2, 'name': 'item2'},
                    ],
                },
            },
            'timestamps': {
                'created': '2023-01-01T00:00:00Z',
                'last_login': '2023-01-02T12:00:00Z',
            },
            'numbers': [1, 2, 3.14, -5],
            'boolean_flags': {'is_active': True, 'is_verified': False},
        }

        # Test JWS round trip
        jws_token = jwt_service.create_jws_token(complex_payload)
        jws_decoded = jwt_service.verify_jws_token(jws_token)

        # Verify complex structure is preserved
        assert jws_decoded['user_id'] == complex_payload['user_id']
        assert (
            jws_decoded['metadata']['permissions']
            == complex_payload['metadata']['permissions']
        )
        assert (
            jws_decoded['metadata']['settings']['nested_array']
            == complex_payload['metadata']['settings']['nested_array']
        )
        assert jws_decoded['numbers'] == complex_payload['numbers']
        assert jws_decoded['boolean_flags'] == complex_payload['boolean_flags']

        # Test JWE round trip
        jwe_token = jwt_service.create_jwe_token(complex_payload)
        jwe_decrypted = jwt_service.decrypt_jwe_token(jwe_token)

        # Verify complex structure is preserved
        assert jwe_decrypted['user_id'] == complex_payload['user_id']
        assert (
            jwe_decrypted['metadata']['permissions']
            == complex_payload['metadata']['permissions']
        )
        assert (
            jwe_decrypted['metadata']['settings']['nested_array']
            == complex_payload['metadata']['settings']['nested_array']
        )
        assert jwe_decrypted['numbers'] == complex_payload['numbers']
        assert jwe_decrypted['boolean_flags'] == complex_payload['boolean_flags']

    @patch('openhands.app_server.services.jwt_service.utc_now')
    def test_token_expiration_timing(self, mock_utc_now, jwt_service):
        """Test that token expiration is set correctly."""
        # Mock the current time
        fixed_time = datetime(2023, 1, 1, 12, 0, 0)
        mock_utc_now.return_value = fixed_time

        payload = {'user_id': '123'}
        expires_in = timedelta(hours=1)

        # Create JWS token
        jws_token = jwt_service.create_jws_token(payload, expires_in=expires_in)

        # Decode without verification to check timestamps (since token is "expired" in real time)
        import jwt as pyjwt

        jws_decoded = pyjwt.decode(
            jws_token, options={'verify_signature': False, 'verify_exp': False}
        )

        # JWT library converts datetime to Unix timestamps
        assert jws_decoded['iat'] == int(fixed_time.timestamp())
        assert jws_decoded['exp'] == int((fixed_time + expires_in).timestamp())

        # Create JWE token
        jwe_token = jwt_service.create_jwe_token(payload, expires_in=expires_in)
        jwe_decrypted = jwt_service.decrypt_jwe_token(jwe_token)

        assert jwe_decrypted['iat'] == int(fixed_time.timestamp())
        assert jwe_decrypted['exp'] == int((fixed_time + expires_in).timestamp())

    def test_empty_payload(self, jwt_service):
        """Test JWS and JWE with empty payload."""
        empty_payload = {}

        # Test JWS
        jws_token = jwt_service.create_jws_token(empty_payload)
        jws_decoded = jwt_service.verify_jws_token(jws_token)

        # Should still have standard claims
        assert 'iat' in jws_decoded
        assert 'exp' in jws_decoded

        # Test JWE
        jwe_token = jwt_service.create_jwe_token(empty_payload)
        jwe_decrypted = jwt_service.decrypt_jwe_token(jwe_token)

        # Should still have standard claims
        assert 'iat' in jwe_decrypted
        assert 'exp' in jwe_decrypted

    def test_unicode_and_special_characters(self, jwt_service):
        """Test JWS and JWE with unicode and special characters."""
        unicode_payload = {
            'user_name': 'José María',
            'description': 'Testing with émojis 🚀 and spëcial chars: @#$%^&*()',
            'chinese': '你好世界',
            'arabic': 'مرحبا بالعالم',
            'symbols': '∑∆∏∫√∞≠≤≥',
        }

        # Test JWS round trip
        jws_token = jwt_service.create_jws_token(unicode_payload)
        jws_decoded = jwt_service.verify_jws_token(jws_token)

        for key, value in unicode_payload.items():
            assert jws_decoded[key] == value

        # Test JWE round trip
        jwe_token = jwt_service.create_jwe_token(unicode_payload)
        jwe_decrypted = jwt_service.decrypt_jwe_token(jwe_token)

        for key, value in unicode_payload.items():
            assert jwe_decrypted[key] == value
