# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
from typing import Any

from litellm import BaseModel

from openhands.resolver.interfaces.issue import Issue


class ResolverOutput(BaseModel):
    # NOTE: User-specified
    issue: Issue
    issue_type: str
    instruction: str
    base_commit: str
    git_patch: str
    history: list[dict[str, Any]]
    metrics: dict[str, Any] | None
    success: bool
    comment_success: list[bool] | None
    result_explanation: str
    error: str | None
