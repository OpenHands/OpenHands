# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
"""Constants for tool names used in function calling."""

EXECUTE_BASH_TOOL_NAME = 'execute_bash'
STR_REPLACE_EDITOR_TOOL_NAME = 'str_replace_editor'
BROWSER_TOOL_NAME = 'browser'
FINISH_TOOL_NAME = 'finish'
LLM_BASED_EDIT_TOOL_NAME = 'edit_file'
TASK_TRACKER_TOOL_NAME = 'task_tracker'
