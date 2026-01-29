# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from typing import Any

from pydantic import RootModel


class ExtendedConfig(RootModel[dict[str, Any]]):
    """Configuration for extended functionalities.

    This is implemented as a root model so that the entire input is stored
    as the root value. This allows arbitrary keys to be stored and later
    accessed via attribute or dictionary-style access.
    """

    def __str__(self) -> str:
        # Use the root dict to build a string representation.
        root_dict: dict[str, Any] = self.model_dump()
        attr_str = [f'{k}={repr(v)}' for k, v in root_dict.items()]
        return f'ExtendedConfig({", ".join(attr_str)})'

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ExtendedConfig':
        # Create an instance directly by wrapping the input dict.
        return cls(data)

    def __getitem__(self, key: str) -> Any:
        # Provide dictionary-like access via the root dict.
        root_dict: dict[str, Any] = self.model_dump()
        return root_dict[key]

    def __getattr__(self, key: str) -> Any:
        # Fallback for attribute access using the root dict.
        try:
            root_dict: dict[str, Any] = self.model_dump()
            return root_dict[key]
        except KeyError as e:
            raise AttributeError(
                f"'ExtendedConfig' object has no attribute '{key}'"
            ) from e
