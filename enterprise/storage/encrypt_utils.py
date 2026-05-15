import binascii
import hashlib
import json
from base64 import b64decode, b64encode
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, SecretStr
from sqlalchemy import String, TypeDecorator
from sqlalchemy.engine.interfaces import Dialect

_jwt_service = None
_fernet = None


def encrypt_value(value: str | SecretStr) -> str:
    raw = value.get_secret_value() if isinstance(value, SecretStr) else value
    return get_jwt_service().encrypt_value(raw)


def decrypt_value(value: str | SecretStr) -> str:
    raw = value.get_secret_value() if isinstance(value, SecretStr) else value
    return get_jwt_service().decrypt_value(raw)


def get_jwt_service():
    from openhands.app_server.config import get_global_config

    global _jwt_service
    if _jwt_service is None:
        jwt_service_injector = get_global_config().jwt
        assert jwt_service_injector is not None
        _jwt_service = jwt_service_injector.get_jwt_service()
    return _jwt_service


def decrypt_legacy_model(decrypt_keys: list, model_instance) -> dict:
    return decrypt_legacy_kwargs(decrypt_keys, model_to_kwargs(model_instance))


def decrypt_legacy_kwargs(encrypt_keys: list, kwargs: dict) -> dict:
    for key, value in kwargs.items():
        try:
            if value is None:
                continue
            if key in encrypt_keys:
                value = decrypt_legacy_value(value)
                kwargs[key] = value
        except binascii.Error:
            pass  # Key is in legacy format...
        except InvalidToken:
            pass  # Key not encrypted...
    return kwargs


def decrypt_legacy_value(value: str | SecretStr) -> str:
    if isinstance(value, SecretStr):
        return (
            get_fernet().decrypt(b64decode(value.get_secret_value().encode())).decode()
        )
    else:
        return get_fernet().decrypt(b64decode(value.encode())).decode()


def encrypt_legacy_value(value: str | SecretStr) -> str:
    if isinstance(value, SecretStr):
        return b64encode(
            get_fernet().encrypt(value.get_secret_value().encode())
        ).decode()
    else:
        return b64encode(get_fernet().encrypt(value.encode())).decode()


def get_fernet():
    global _fernet
    if _fernet is None:
        jwt_svc = get_jwt_service()
        default_key = jwt_svc.get_key(jwt_svc._default_key_id)
        secret = default_key.key.get_secret_value()
        fernet_key = b64encode(hashlib.sha256(secret.encode()).digest())
        _fernet = Fernet(fernet_key)
    return _fernet


def model_to_kwargs(model_instance):
    return {
        column.name: getattr(model_instance, column.name)
        for column in model_instance.__table__.columns
    }


class EncryptedJSON(TypeDecorator[dict[str, Any]]):
    """JSON column whose serialized payload is encrypted at rest.

    Accepts either a plain ``dict`` or a pydantic ``BaseModel``. Pydantic
    models are dumped via ``model_dump(mode='json', context={'expose_secrets': True})``
    so nested ``SecretStr`` values keep their real payload — the column
    itself is the encryption boundary, so masking on the way in would
    corrupt round-trips.

    Use for JSON payloads that may contain secrets (e.g. nested ``api_key``
    fields) where the existing ``_<field>`` String + property pattern is
    awkward — this keeps the column accessible as a normal ORM attribute
    while encrypting the entire JSON blob via the same JWE service used
    by ``encrypt_value``/``decrypt_value``.
    """

    impl = String
    cache_ok = True

    def process_bind_param(
        self, value: BaseModel | dict[str, Any] | None, dialect: Dialect
    ) -> str | None:
        if value is None:
            return None
        if isinstance(value, BaseModel):
            value = value.model_dump(mode='json', context={'expose_secrets': True})
        return encrypt_value(json.dumps(value))

    def process_result_value(
        self, value: str | None, dialect: Dialect
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        return json.loads(decrypt_value(value))
