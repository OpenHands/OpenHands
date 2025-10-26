import binascii
import hashlib
from base64 import b64decode, b64encode

from cryptography.fernet import Fernet
from pydantic import SecretStr
from server.config import get_config

_fernet = None


def encrypt_model(encrypt_keys: list, model_instance) -> dict:
    return encrypt_kwargs(encrypt_keys, model_to_kwargs(model_instance))


def decrypt_model(decrypt_keys: list, model_instance) -> dict:
    return decrypt_kwargs(decrypt_keys, model_to_kwargs(model_instance))


def encrypt_kwargs(encrypt_keys: list, kwargs: dict) -> dict:
    fernet = get_fernet()
    for key, value in kwargs.items():
        if value is None:
            continue

        if isinstance(value, dict):
            encrypt_kwargs(encrypt_keys, value)
            continue

        if key in encrypt_keys:
            if isinstance(value, SecretStr):
                value = b64encode(
                    fernet.encrypt(value.get_secret_value().encode())
                ).decode()
            else:
                value = b64encode(fernet.encrypt(value.encode())).decode()
            kwargs[key] = value
    return kwargs


def decrypt_kwargs(encrypt_keys: list, kwargs: dict) -> dict:
    fernet = get_fernet()
    for key, value in kwargs.items():
        try:
            if value is None:
                continue
            if key in encrypt_keys:
                if isinstance(value, SecretStr):
                    value = fernet.decrypt(
                        b64decode(value.get_secret_value().encode())
                    ).decode()
                else:
                    value = fernet.decrypt(b64decode(value.encode())).decode()
                kwargs[key] = value
        except binascii.Error:
            pass  # Key is in legacy format...
    return kwargs


def encrypt_value(value: str | SecretStr) -> str:
    if isinstance(value, SecretStr):
        return b64encode(
            get_fernet().encrypt(value.get_secret_value().encode())
        ).decode()
    else:
        return b64encode(get_fernet().encrypt(value.encode())).decode()


def decrypt_value(value: str | SecretStr) -> str:
    if isinstance(value, SecretStr):
        return (
            get_fernet().decrypt(b64decode(value.get_secret_value().encode())).decode()
        )
    else:
        return get_fernet().decrypt(b64decode(value.encode())).decode()


def get_fernet():
    global _fernet
    if _fernet is None:
        jwt_secret = get_config().jwt_secret.get_secret_value()
        fernet_key = b64encode(hashlib.sha256(jwt_secret.encode()).digest())
        _fernet = Fernet(fernet_key)
    return _fernet


def model_to_kwargs(model_instance):
    return {
        column.name: getattr(model_instance, column.name)
        for column in model_instance.__table__.columns
    }
