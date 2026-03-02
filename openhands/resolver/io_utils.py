# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import json
from typing import Iterable

from openhands.resolver.resolver_output import ResolverOutput


def load_all_resolver_outputs(output_jsonl: str) -> Iterable[ResolverOutput]:
    with open(output_jsonl, 'r') as f:
        for line in f:
            yield ResolverOutput.model_validate(json.loads(line))


def load_single_resolver_output(output_jsonl: str, issue_number: int) -> ResolverOutput:
    for resolver_output in load_all_resolver_outputs(output_jsonl):
        if resolver_output.issue.number == issue_number:
            return resolver_output
    raise ValueError(f'Issue number {issue_number} not found in {output_jsonl}')
