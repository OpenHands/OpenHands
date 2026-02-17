# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import os

from openai import OpenAI


# ==================================================================================================
# OPENAI
# TODO: Move this to EventStream Actions when DockerRuntime is fully implemented
# NOTE: we need to get env vars inside functions because they will be set in IPython
# AFTER the agentskills is imported (the case for DockerRuntime)
# ==================================================================================================
def _get_openai_api_key() -> str:
    return os.getenv('OPENAI_API_KEY', os.getenv('SANDBOX_ENV_OPENAI_API_KEY', ''))


def _get_openai_base_url() -> str:
    return os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')


def _get_openai_model() -> str:
    return os.getenv('OPENAI_MODEL', 'gpt-4o')


def _get_max_token() -> int:
    return int(os.getenv('MAX_TOKEN', '500'))


def _get_openai_client() -> OpenAI:
    client = OpenAI(api_key=_get_openai_api_key(), base_url=_get_openai_base_url())
    return client
