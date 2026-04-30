# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import argparse
import os
import pathlib
import sys
from ast import literal_eval
from types import UnionType
from typing import Any, MutableMapping, get_args, get_origin, get_type_hints
from uuid import uuid4

import toml
from dotenv import load_dotenv
from pydantic import BaseModel, SecretStr, ValidationError

from openhands.app_server.file_store import get_file_store
from openhands.app_server.file_store.files import FileStore
from openhands.app_server.utils import logger
from openhands.core.config.arg_utils import get_headless_parser
from openhands.core.config.mcp_config import mcp_config_from_toml
from openhands.core.config.openhands_config import OpenHandsConfig

JWT_SECRET = '.jwt_secret'
load_dotenv()


def load_from_env(
    cfg: OpenHandsConfig, env_or_toml_dict: dict | MutableMapping[str, str]
) -> None:
    """Sets config attributes from environment variables or TOML dictionary.

    Reads environment-style variables and updates the config attributes accordingly.
    Supports configuration of LLM settings (e.g., LLM_BASE_URL), agent settings
    (e.g., AGENT_MEMORY_ENABLED), sandbox settings (e.g., SANDBOX_TIMEOUT), and more.

    Args:
        cfg: The OpenHandsConfig object to set attributes on.
        env_or_toml_dict: The environment variables or a config.toml dict.
    """

    def get_optional_type(union_type: UnionType | type | None) -> type | None:
        """Returns the non-None type from a Union."""
        if union_type is None:
            return None
        if get_origin(union_type) is UnionType:
            types = get_args(union_type)
            return next((t for t in types if t is not type(None)), None)
        if isinstance(union_type, type):
            return union_type
        return None

    # helper function to set attributes based on env vars
    def set_attr_from_env(sub_config: BaseModel, prefix: str = '') -> None:
        """Set attributes of a config model based on environment variables."""
        for field_name, field_info in sub_config.__class__.model_fields.items():
            field_value = getattr(sub_config, field_name)
            field_type = field_info.annotation

            # compute the expected env var name from the prefix and field name
            # e.g. LLM_BASE_URL
            env_var_name = (prefix + field_name).upper()

            cast_value: Any
            if isinstance(field_value, BaseModel):
                set_attr_from_env(field_value, prefix=field_name + '_')

            elif env_var_name in env_or_toml_dict:
                # convert the env var to the correct type and set it
                value = env_or_toml_dict[env_var_name]

                # skip empty config values (fall back to default)
                if not value:
                    continue

                try:
                    # if it's an optional type, get the non-None type
                    if get_origin(field_type) is UnionType:
                        field_type = get_optional_type(field_type)

                    # Attempt to cast the env var to type hinted in the dataclass
                    if field_type is bool:
                        cast_value = str(value).lower() in ['true', '1']
                    # parse dicts and lists like SANDBOX_RUNTIME_STARTUP_ENV_VARS and SANDBOX_RUNTIME_EXTRA_BUILD_ARGS
                    elif (
                        get_origin(field_type) is dict
                        or get_origin(field_type) is list
                        or field_type is dict
                        or field_type is list
                    ):
                        cast_value = literal_eval(value)
                        # If it's a list of Pydantic models
                        if get_origin(field_type) is list:
                            inner_type = get_args(field_type)[0]
                            if isinstance(inner_type, type) and issubclass(
                                inner_type, BaseModel
                            ):
                                cast_value = [
                                    inner_type(**item)
                                    if isinstance(item, dict)
                                    else item
                                    for item in cast_value
                                ]
                    else:
                        if field_type is not None:
                            cast_value = field_type(value)
                    setattr(sub_config, field_name, cast_value)
                except (ValueError, TypeError):
                    logger.openhands_logger.error(
                        f'Error setting env var {env_var_name}={value}: check that the value is of the right type'
                    )

    # Start processing from the root of the config object
    set_attr_from_env(cfg)


def load_from_toml(cfg: OpenHandsConfig, toml_file: str = 'config.toml') -> None:
    """Load the config from the toml file. Supports both styles of config vars.

    Args:
        cfg: The OpenHandsConfig object to update attributes of.
        toml_file: The path to the toml file. Defaults to 'config.toml'.

    See Also:
    - config.template.toml for the full list of config options.
    """
    # try to read the config.toml file into the config object
    try:
        with open(toml_file, 'r', encoding='utf-8') as toml_contents:
            toml_config = toml.load(toml_contents)
    except FileNotFoundError as e:
        logger.openhands_logger.info(
            f'{toml_file} not found: {e}. Toml values have not been applied.'
        )
        return
    except toml.TomlDecodeError as e:
        logger.openhands_logger.warning(
            f'Cannot parse config from toml, toml values have not been applied.\nError: {e}',
        )
        return

    # Check for the [core] section
    if 'core' not in toml_config:
        logger.openhands_logger.warning(
            f'No [core] section found in {toml_file}. Core settings will use defaults.'
        )
        core_config = {}
    else:
        core_config = toml_config['core']

    # Process core section if present
    cfg_type_hints = get_type_hints(cfg.__class__)
    for key, value in core_config.items():
        if hasattr(cfg, key):
            # Get expected type of the attribute
            expected_type = cfg_type_hints.get(key, None)

            # Check if expected_type is a Union that includes SecretStr and value is str, e.g. search_api_key
            if expected_type:
                origin = get_origin(expected_type)
                args = get_args(expected_type)

                if origin is UnionType and SecretStr in args and isinstance(value, str):
                    value = SecretStr(value)
                elif expected_type is SecretStr and isinstance(value, str):
                    value = SecretStr(value)

            setattr(cfg, key, value)
        else:
            logger.openhands_logger.warning(
                f'Unknown config key "{key}" in [core] section'
            )

    # Process MCP sections if present
    if 'mcp' in toml_config:
        try:
            mcp_mapping = mcp_config_from_toml(toml_config['mcp'])
            if 'mcp' in mcp_mapping:
                cfg.mcp = mcp_mapping['mcp']
        except (TypeError, KeyError, ValidationError) as e:
            logger.openhands_logger.warning(
                f'Cannot parse MCP config from toml, values have not been applied.\nError: {e}'
            )
        except ValueError:
            raise ValueError('Error in MCP sections in config.toml')

    # Check for unknown sections
    # Note: legacy sections are kept for backwards compatibility with old config
    # files - they are silently ignored
    known_sections = {
        'core',
        'mcp',
        'llm',  # Legacy, ignored
        'sandbox',  # Legacy, ignored
        'security',  # Legacy, ignored
        'agent',  # Legacy, ignored
        'extended',  # Legacy, ignored
        'condenser',  # Legacy, ignored
        'kubernetes',  # Legacy, ignored
        'model_routing',  # Legacy, ignored
    }
    for key in toml_config:
        if key.lower() not in known_sections:
            logger.openhands_logger.warning(f'Unknown section [{key}] in {toml_file}')


def get_or_create_jwt_secret(file_store: FileStore) -> str:
    try:
        jwt_secret = file_store.read(JWT_SECRET)
        return jwt_secret
    except FileNotFoundError:
        new_secret = uuid4().hex
        file_store.write(JWT_SECRET, new_secret)
        return new_secret


def finalize_config(cfg: OpenHandsConfig) -> None:
    """More tweaks to the config after it's been loaded."""
    # Handle the deprecated workspace_* parameters
    if cfg.workspace_base is not None or cfg.workspace_mount_path is not None:
        if cfg.workspace_base is not None:
            cfg.workspace_base = os.path.abspath(cfg.workspace_base)
            if cfg.workspace_mount_path is None:
                cfg.workspace_mount_path = cfg.workspace_base

        if cfg.workspace_mount_rewrite:
            base = cfg.workspace_base or os.getcwd()
            parts = cfg.workspace_mount_rewrite.split(':')
            cfg.workspace_mount_path = base.replace(parts[0], parts[1])

    # make sure cache dir exists
    if cfg.cache_dir:
        pathlib.Path(cfg.cache_dir).mkdir(parents=True, exist_ok=True)

    if not cfg.jwt_secret:
        cfg.jwt_secret = SecretStr(
            get_or_create_jwt_secret(
                get_file_store(cfg.file_store, cfg.file_store_path)
            )
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = get_headless_parser()
    args = parser.parse_args()
    from openhands import get_version

    if args.version:
        print(f'OpenHands version: {get_version()}')
        sys.exit(0)

    return args


def load_openhands_config(
    set_logging_levels: bool = True, config_file: str = 'config.toml'
) -> OpenHandsConfig:
    """Load the configuration from the specified config file and environment variables.

    Args:
        set_logging_levels: Whether to set the global variables for logging levels.
        config_file: Path to the config file. Defaults to 'config.toml' in the current directory.
    """
    config = OpenHandsConfig()
    load_from_toml(config, config_file)
    load_from_env(config, os.environ)
    finalize_config(config)
    if set_logging_levels:
        logger.DEBUG = config.debug
        logger.DISABLE_COLOR_PRINTING = config.disable_color
    return config


def setup_config_from_args(args: argparse.Namespace) -> OpenHandsConfig:
    """Load config from toml and override with command line arguments.

    Common setup used by both CLI and main.py entry points.

    Configuration precedence (from highest to lowest):
    1. CLI parameters
    2. config.toml in current directory (or --config-file location if specified)
    3. ~/.openhands/settings.json and ~/.openhands/config.toml
    """
    # Load base config from toml and env vars
    config = load_openhands_config(config_file=args.config_file)

    # Override default agent if provided
    if hasattr(args, 'agent_cls') and args.agent_cls:
        config.default_agent = args.agent_cls

    # Set max iterations and max budget per task if provided, otherwise fall back to config values
    if hasattr(args, 'max_iterations') and args.max_iterations is not None:
        config.max_iterations = args.max_iterations
    if hasattr(args, 'max_budget_per_task') and args.max_budget_per_task is not None:
        config.max_budget_per_task = args.max_budget_per_task

    return config
