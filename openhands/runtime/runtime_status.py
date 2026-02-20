# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from enum import Enum


class RuntimeStatus(Enum):
    STOPPED = 'STATUS$STOPPED'
    BUILDING_RUNTIME = 'STATUS$BUILDING_RUNTIME'
    STARTING_RUNTIME = 'STATUS$STARTING_RUNTIME'
    RUNTIME_STARTED = 'STATUS$RUNTIME_STARTED'
    SETTING_UP_WORKSPACE = 'STATUS$SETTING_UP_WORKSPACE'
    SETTING_UP_GIT_HOOKS = 'STATUS$SETTING_UP_GIT_HOOKS'
    READY = 'STATUS$READY'
    ERROR = 'STATUS$ERROR'
    ERROR_RUNTIME_DISCONNECTED = 'STATUS$ERROR_RUNTIME_DISCONNECTED'
    ERROR_LLM_AUTHENTICATION = 'STATUS$ERROR_LLM_AUTHENTICATION'
    ERROR_LLM_SERVICE_UNAVAILABLE = 'STATUS$ERROR_LLM_SERVICE_UNAVAILABLE'
    ERROR_LLM_INTERNAL_SERVER_ERROR = 'STATUS$ERROR_LLM_INTERNAL_SERVER_ERROR'
    ERROR_LLM_OUT_OF_CREDITS = 'STATUS$ERROR_LLM_OUT_OF_CREDITS'
    ERROR_LLM_CONTENT_POLICY_VIOLATION = 'STATUS$ERROR_LLM_CONTENT_POLICY_VIOLATION'
    AGENT_RATE_LIMITED_STOPPED_MESSAGE = (
        'CHAT_INTERFACE$AGENT_RATE_LIMITED_STOPPED_MESSAGE'
    )
    GIT_PROVIDER_AUTHENTICATION_ERROR = 'STATUS$GIT_PROVIDER_AUTHENTICATION_ERROR'
    LLM_RETRY = 'STATUS$LLM_RETRY'
    ERROR_MEMORY = 'STATUS$ERROR_MEMORY'
