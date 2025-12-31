from datetime import UTC, datetime
from enum import Enum
from typing import TypeVar

from pydantic import SecretStr, TypeAdapter
from sqlalchemy import JSON, DateTime, String, TypeDecorator
from sqlalchemy.orm import declarative_base

Base = declarative_base()
T = TypeVar('T', bound=Enum)


def create_json_type_decorator(object_type: type):
    """Create a decorator for a particular type. Introduced because SQLAlchemy could not process lists of enum values."""
    type_adapter: TypeAdapter = TypeAdapter(object_type)

    class JsonTypeDecorator(TypeDecorator):
        impl = JSON
        cache_ok = True

        def process_bind_param(self, value, dialect):
            return type_adapter.dump_python(
                value, mode='json', context={'expose_secrets': True}
            )

        def process_result_param(self, value, dialect):
            return type_adapter.validate_python(value)

    return JsonTypeDecorator


class StoredSecretStr(TypeDecorator):
    """TypeDecorator for secret strings. Encrypts the value using the default key before storing."""

    impl = String
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None:
            from openhands.app_server.config import get_global_config

            jwt_service_injector = get_global_config().jwt
            assert jwt_service_injector is not None
            jwt_service = jwt_service_injector.get_jwt_service()
            token = jwt_service.create_jwe_token({'v': value.get_secret_value()})
            return token
        return None

    def process_result_param(self, value, dialect):
        if value is not None:
            from openhands.app_server.config import get_global_config

            jwt_service_injector = get_global_config().jwt
            assert jwt_service_injector is not None
            jwt_service = jwt_service_injector.get_jwt_service()
            token = jwt_service.decrypt_jwe_token(value)
            return SecretStr(token['v'])
        return None


class UtcDateTime(TypeDecorator):
    """TypeDecorator for datetime - stores all datetimes in utc. Assumes datetime without
    a specified timezone are utc. (Sqlite doesn't always return these)"""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if isinstance(value, datetime) and value.tzinfo != UTC:
            value = value.astimezone(UTC)
        return value

    def process_result_param(self, value, dialect):
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            elif value.tzinfo != UTC:
                value = value.astimezone(UTC)
        return value


def create_enum_type_decorator(enum_type: type[T]):
    class EnumTypeDecorator(TypeDecorator):
        impl = String
        cache_ok = True

        def process_bind_param(self, value, dialect):
            if value is None:
                return None
            return value.value

        def process_result_param(self, value, dialect):
            if value:
                return enum_type[value]

    return EnumTypeDecorator


def row2dict(row):
    d = {}
    for column in row.__table__.columns:
        d[column.name] = getattr(row, column.name)

    return d
