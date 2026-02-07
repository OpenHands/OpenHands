import base64
import hashlib
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Any, AsyncGenerator

import jwt
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import Request
from pydantic import BaseModel, PrivateAttr

from openhands.agent_server.utils import utc_now
from openhands.app_server.services.injector import Injector, InjectorState
from openhands.app_server.utils.encryption_key import (
    EncryptionKey,
    get_default_encryption_keys,
)


def _b64url_encode(data: bytes) -> str:
    """Base64url-encode bytes without padding (per RFC 7515)."""
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('ascii')


def _b64url_decode(s: str) -> bytes:
    """Base64url-decode a string, adding back padding as needed."""
    # Add padding
    padding = 4 - len(s) % 4
    if padding != 4:
        s += '=' * padding
    return base64.urlsafe_b64decode(s)


def _jwe_encrypt(plaintext: bytes, key: bytes, kid: str) -> str:
    """Create a JWE compact token using dir + A256GCM.

    Implements RFC 7516 JWE Compact Serialization with:
    - alg: dir (direct key agreement, no key wrapping)
    - enc: A256GCM (AES-256-GCM content encryption)

    Args:
        plaintext: The data to encrypt.
        key: 256-bit (32-byte) encryption key.
        kid: Key ID to include in the JWE header.

    Returns:
        JWE compact serialization string.
    """
    # Build the protected header
    header = {'alg': 'dir', 'enc': 'A256GCM', 'kid': kid}
    header_json = json.dumps(header, separators=(',', ':'), sort_keys=False)
    protected_header = _b64url_encode(header_json.encode('utf-8'))

    # For "dir" algorithm, the encrypted key is empty
    encrypted_key = ''

    # Generate a random 96-bit (12-byte) IV for AES-256-GCM
    iv = os.urandom(12)
    iv_b64 = _b64url_encode(iv)

    # AAD is the ASCII bytes of the base64url-encoded protected header (RFC 7516 sec 5.1)
    aad = protected_header.encode('ascii')

    # Encrypt using AES-256-GCM
    aesgcm = AESGCM(key)
    ciphertext_and_tag = aesgcm.encrypt(iv, plaintext, aad)

    # AES-GCM appends the 16-byte tag to the ciphertext
    ciphertext = ciphertext_and_tag[:-16]
    tag = ciphertext_and_tag[-16:]

    ciphertext_b64 = _b64url_encode(ciphertext)
    tag_b64 = _b64url_encode(tag)

    # JWE Compact Serialization: Header.EncryptedKey.IV.Ciphertext.Tag
    return f'{protected_header}.{encrypted_key}.{iv_b64}.{ciphertext_b64}.{tag_b64}'


def _jwe_get_header(token: str) -> dict[str, Any]:
    """Extract and decode the unverified header from a JWE compact token.

    Args:
        token: JWE compact serialization string.

    Returns:
        The decoded header dictionary.

    Raises:
        ValueError: If the token format is invalid.
    """
    parts = token.split('.')
    if len(parts) != 5:
        raise ValueError('Invalid JWE token format: expected 5 parts')
    return json.loads(_b64url_decode(parts[0]))


def _jwe_decrypt(token: str, key: bytes) -> bytes:
    """Decrypt a JWE compact token using dir + A256GCM.

    Args:
        token: JWE compact serialization string.
        key: 256-bit (32-byte) decryption key.

    Returns:
        The decrypted plaintext bytes.

    Raises:
        ValueError: If the token format is invalid.
        Exception: If decryption fails.
    """
    parts = token.split('.')
    if len(parts) != 5:
        raise ValueError('Invalid JWE token format: expected 5 parts')

    protected_header_b64, _encrypted_key_b64, iv_b64, ciphertext_b64, tag_b64 = parts

    iv = _b64url_decode(iv_b64)
    ciphertext = _b64url_decode(ciphertext_b64)
    tag = _b64url_decode(tag_b64)

    # AAD is the ASCII bytes of the base64url-encoded protected header
    aad = protected_header_b64.encode('ascii')

    # Reconstruct ciphertext + tag for AESGCM.decrypt
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(iv, ciphertext + tag, aad)
    return plaintext


class JwtService:
    """Service for signing/verifying JWS tokens and encrypting/decrypting JWE tokens."""

    def __init__(self, keys: list[EncryptionKey]):
        """Initialize the JWT service with a list of keys.

        Args:
            keys: List of EncryptionKey objects. If None, will try to load from config.

        Raises:
            ValueError: If no keys are provided and config is not available
        """
        active_keys = [key for key in keys if key.active]
        if not active_keys:
            raise ValueError('At least one active key is required')

        # Store keys by ID for quick lookup
        self._keys = {key.id: key for key in keys}

        # Find the newest key as default
        newest_key = max(active_keys, key=lambda k: k.created_at)
        self._default_key_id = newest_key.id

    @property
    def default_key_id(self) -> str:
        """Get the default key ID."""
        return self._default_key_id

    def create_jws_token(
        self,
        payload: dict[str, Any],
        key_id: str | None = None,
        expires_in: timedelta | None = None,
    ) -> str:
        """Create a JWS (JSON Web Signature) token.

        Args:
            payload: The JWT payload
            key_id: The key ID to use for signing. If None, uses the newest key.
            expires_in: Token expiration time. If None, defaults to 1 hour.

        Returns:
            The signed JWS token

        Raises:
            ValueError: If key_id is invalid
        """
        if key_id is None:
            key_id = self._default_key_id

        if key_id not in self._keys:
            raise ValueError(f"Key ID '{key_id}' not found")

        # Add standard JWT claims
        now = utc_now()
        if expires_in is None:
            expires_in = timedelta(hours=1)

        jwt_payload = {
            **payload,
            'iat': int(now.timestamp()),
            'exp': int((now + expires_in).timestamp()),
        }

        # Use the raw key for JWT signing with key_id in header
        secret_key = self._keys[key_id].key.get_secret_value()

        return jwt.encode(
            jwt_payload, secret_key, algorithm='HS256', headers={'kid': key_id}
        )

    def verify_jws_token(self, token: str, key_id: str | None = None) -> dict[str, Any]:
        """Verify and decode a JWS token.

        Args:
            token: The JWS token to verify
            key_id: The key ID to use for verification. If None, extracts from
                    token's kid header.

        Returns:
            The decoded JWT payload

        Raises:
            ValueError: If token is invalid or key_id is not found
            jwt.InvalidTokenError: If token verification fails
        """
        if key_id is None:
            # Try to extract key_id from the token's kid header
            try:
                unverified_header = jwt.get_unverified_header(token)
                key_id = unverified_header.get('kid')
                if not key_id:
                    raise ValueError("Token does not contain 'kid' header with key ID")
            except jwt.DecodeError:
                raise ValueError('Invalid JWT token format')

        if key_id not in self._keys:
            raise ValueError(f"Key ID '{key_id}' not found")

        # Use the raw key for JWT verification
        secret_key = self._keys[key_id].key.get_secret_value()

        try:
            payload = jwt.decode(token, secret_key, algorithms=['HS256'])
            return payload
        except jwt.InvalidTokenError as e:
            raise jwt.InvalidTokenError(f'Token verification failed: {str(e)}')

    def create_jwe_token(
        self,
        payload: dict[str, Any],
        key_id: str | None = None,
        expires_in: timedelta | None = None,
    ) -> str:
        """Create a JWE (JSON Web Encryption) token.

        Uses dir + A256GCM via the cryptography library, replacing python-jose
        to eliminate known token/algorithm confusion vulnerabilities.

        Args:
            payload: The JWT payload to encrypt
            key_id: The key ID to use for encryption. If None, uses the newest key.
            expires_in: Token expiration time. If None, defaults to 1 hour.

        Returns:
            The encrypted JWE token

        Raises:
            ValueError: If key_id is invalid
        """
        if key_id is None:
            key_id = self._default_key_id

        if key_id not in self._keys:
            raise ValueError(f"Key ID '{key_id}' not found")

        # Add standard JWT claims
        now = utc_now()
        jwt_payload = {
            **payload,
            'iat': int(now.timestamp()),
        }

        # Only add exp if expires_in is provided
        if expires_in is not None:
            jwt_payload['exp'] = int((now + expires_in).timestamp())

        # Get the raw key for JWE encryption and derive a 256-bit key
        secret_key = self._keys[key_id].key.get_secret_value()
        key_bytes = secret_key.encode() if isinstance(secret_key, str) else secret_key
        # Derive a 256-bit key using SHA256
        key_256 = hashlib.sha256(key_bytes).digest()

        # Encrypt the payload
        payload_json = json.dumps(jwt_payload)
        return _jwe_encrypt(payload_json.encode('utf-8'), key_256, kid=key_id)

    def decrypt_jwe_token(
        self, token: str, key_id: str | None = None
    ) -> dict[str, Any]:
        """Decrypt and decode a JWE token.

        Args:
            token: The JWE token to decrypt
            key_id: The key ID to use for decryption. If None, extracts
                    from token header.

        Returns:
            The decrypted JWT payload

        Raises:
            ValueError: If token is invalid or key_id is not found
            Exception: If token decryption fails
        """
        if key_id is None:
            # Try to extract key_id from the token's header
            try:
                header = _jwe_get_header(token)
                key_id = header.get('kid')
                if not key_id:
                    raise ValueError("Token does not contain 'kid' header with key ID")
            except Exception:
                raise ValueError('Invalid JWE token format')

        if key_id not in self._keys:
            raise ValueError(f"Key ID '{key_id}' not found")

        # Get the raw key for JWE decryption and derive a 256-bit key
        secret_key = self._keys[key_id].key.get_secret_value()
        key_bytes = secret_key.encode() if isinstance(secret_key, str) else secret_key
        # Derive a 256-bit key using SHA256
        key_256 = hashlib.sha256(key_bytes).digest()

        try:
            plaintext = _jwe_decrypt(token, key_256)
            # Parse the JSON string back to dictionary
            payload = json.loads(plaintext)
            return payload
        except Exception as e:
            raise Exception(f'Token decryption failed: {str(e)}')


class JwtServiceInjector(BaseModel, Injector[JwtService]):
    persistence_dir: Path
    _jwt_service: JwtService | None = PrivateAttr(default=None)

    def get_jwt_service(self) -> JwtService:
        jwt_service = self._jwt_service
        if jwt_service is None:
            keys = get_default_encryption_keys(self.persistence_dir)
            jwt_service = JwtService(keys=keys)
            self._jwt_service = jwt_service
        return jwt_service

    async def inject(
        self, state: InjectorState, request: Request | None = None
    ) -> AsyncGenerator[JwtService, None]:
        yield self.get_jwt_service()
