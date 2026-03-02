# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import argparse
import os

from openhands.resolver.io_utils import load_single_resolver_output


def visualize_resolver_output(
    issue_number: int, output_dir: str, vis_method: str
) -> None:
    output_jsonl = os.path.join(output_dir, 'output.jsonl')
    resolver_output = load_single_resolver_output(output_jsonl, issue_number)
    if vis_method == 'json':
        print(resolver_output.model_dump_json(indent=4))
    else:
        raise ValueError(f'Invalid visualization method: {vis_method}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize a patch.')
    parser.add_argument(
        '--issue-number',
        type=int,
        required=True,
        help='Issue number to send the pull request for.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory to write the results.',
    )
    parser.add_argument(
        '--vis-method',
        type=str,
        default='json',
        choices=['json'],
        help='Method to visualize the patch [json].',
    )
    my_args = parser.parse_args()

    visualize_resolver_output(
        issue_number=my_args.issue_number,
        output_dir=my_args.output_dir,
        vis_method=my_args.vis_method,
    )
