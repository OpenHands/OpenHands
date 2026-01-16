from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic.json import pydantic_encoder

from openhands.integrations.provider import (
    CUSTOM_SECRETS_TYPE,
    PROVIDER_TOKEN_TYPE,
    CustomSecret,
    ProviderToken,
)
from openhands.integrations.service_types import ProviderType


class Secrets(BaseModel):
    provider_tokens: PROVIDER_TOKEN_TYPE = Field(
        default_factory=lambda: MappingProxyType({})
    )

    custom_secrets: CUSTOM_SECRETS_TYPE = Field(
        default_factory=lambda: MappingProxyType({})
    )

    model_config = ConfigDict(
        frozen=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    @field_validator('provider_tokens', 'custom_secrets')
    @classmethod
    def immutable_validator(cls, value: Mapping) -> MappingProxyType:
        return MappingProxyType(value)

    @field_serializer('provider_tokens')
    def provider_tokens_serializer(
        self, provider_tokens: PROVIDER_TOKEN_TYPE, info: SerializationInfo
    ) -> dict[str, dict[str, str | Any]]:
        tokens = {}
        expose_secrets = info.context and info.context.get('expose_secrets', False)

        for token_type, provider_token in provider_tokens.items():
            if not provider_token or not provider_token.token:
                continue

            token_type_str = (
                token_type.value
                if isinstance(token_type, ProviderType)
                else str(token_type)
            )

            token = None
            if provider_token.token:
                token = (
                    provider_token.token.get_secret_value()
                    if expose_secrets
                    else pydantic_encoder(provider_token.token)
                )

            tokens[token_type_str] = {
                'token': token,
                'host': provider_token.host,
                'user_id': provider_token.user_id,
            }

        return tokens

    @field_serializer('custom_secrets')
    def custom_secrets_serializer(
        self, custom_secrets: CUSTOM_SECRETS_TYPE, info: SerializationInfo
    ):
        secrets = {}
        expose_secrets = info.context and info.context.get('expose_secrets', False)

        if custom_secrets:
            for secret_name, secret_value in custom_secrets.items():
                secrets[secret_name] = {
                    'secret': secret_value.secret.get_secret_value()
                    if expose_secrets
                    else pydantic_encoder(secret_value.secret),
                    'description': secret_value.description,
                }

        return secrets

    @model_validator(mode='before')
    @classmethod
    def convert_dict_to_mappingproxy(
        cls, data: dict[str, dict[str, Any] | MappingProxyType] | PROVIDER_TOKEN_TYPE
    ) -> dict[str, MappingProxyType | None]:
        """Custom deserializer to convert dictionary into MappingProxyType"""
        if not isinstance(data, dict):
            raise ValueError('Secrets must be initialized with a dictionary')

        new_data: dict[str, MappingProxyType | None] = {}

        if 'provider_tokens' in data:
            tokens = data['provider_tokens']
            if isinstance(
                tokens, dict
            ):  # Ensure conversion happens only for dict inputs
                converted_tokens = {}
                for key, value in tokens.items():
                    try:
                        provider_type = (
                            ProviderType(key) if isinstance(key, str) else key
                        )
                        converted_tokens[provider_type] = ProviderToken.from_value(
                            value
                        )
                    except ValueError:
                        # Skip invalid provider types or tokens
                        continue

                # Convert to MappingProxyType
                new_data['provider_tokens'] = MappingProxyType(converted_tokens)
            elif isinstance(tokens, MappingProxyType):
                new_data['provider_tokens'] = tokens

        if 'custom_secrets' in data:
            secrets = data['custom_secrets']
            if isinstance(secrets, dict):
                converted_secrets = {}
                for key, value in secrets.items():
                    try:
                        converted_secrets[key] = CustomSecret.from_value(value)
                    except ValueError:
                        continue

                new_data['custom_secrets'] = MappingProxyType(converted_secrets)
            elif isinstance(secrets, MappingProxyType):
                new_data['custom_secrets'] = secrets

        return new_data

    def get_env_vars(self) -> dict[str, str]:
        secret_store = self.model_dump(context={'expose_secrets': True})
        custom_secrets = secret_store.get('custom_secrets', {})
        secrets = {}
        for secret_name, value in custom_secrets.items():
            secrets[secret_name] = value['secret']

        return secrets

    def get_custom_secrets_descriptions(self) -> dict[str, str]:
        secrets = {}
        for secret_name, secret in self.custom_secrets.items():
            secrets[secret_name] = secret.description

        return secrets
