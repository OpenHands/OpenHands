"""String utility functions that don't depend on other OpenHands modules."""

import re
import sys


def refine_prompt(prompt: str) -> str:
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
