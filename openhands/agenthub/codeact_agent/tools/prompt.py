# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import re
import sys


def refine_prompt(prompt: str):
    """Refines the prompt based on the platform.

    On Windows systems, replaces 'bash' with 'powershell' and 'execute_bash' with 'execute_powershell'
    to ensure commands work correctly on the Windows platform.

    Args:
        prompt: The prompt text to refine

    Returns:
        The refined prompt text
    """
    if sys.platform == 'win32':
        # Replace 'bash' with 'powershell' including tool names like 'execute_bash'
        # First replace 'execute_bash' with 'execute_powershell' to handle tool names
        result = re.sub(
            r'\bexecute_bash\b', 'execute_powershell', prompt, flags=re.IGNORECASE
        )
        # Then replace standalone 'bash' with 'powershell'
        result = re.sub(
            r'(?<!execute_)(?<!_)\bbash\b', 'powershell', result, flags=re.IGNORECASE
        )
        return result
    return prompt
