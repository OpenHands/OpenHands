# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import logging
import multiprocessing as mp
import os
import re
from typing import Callable

from pydantic import SecretStr

from openhands.controller.state.state import State
from openhands.core.logger import get_console_handler
from openhands.core.logger import openhands_logger as logger
from openhands.events.action import Action
from openhands.events.action.message import MessageAction
from openhands.integrations.service_types import ProviderType
from openhands.integrations.utils import validate_provider_token


async def identify_token(token: str, base_domain: str | None) -> ProviderType:
    """Identifies whether a token belongs to GitHub, GitLab, Bitbucket, or Azure DevOps.
    Parameters:
        token (str): The personal access token to check.
        base_domain (str): Custom base domain for provider (e.g GitHub Enterprise)
    """
    provider = await validate_provider_token(SecretStr(token), base_domain)
    if not provider:
        raise ValueError('Token is invalid.')

    return provider


def codeact_user_response(
    state: State,
    encapsulate_solution: bool = False,
    try_parse: Callable[[Action | None], str] | None = None,
) -> str:
    encaps_str = (
        (
            'Please encapsulate your final answer (answer ONLY) within <solution> and </solution>.\n'
            'For example: The answer to the question is <solution> 42 </solution>.\n'
        )
        if encapsulate_solution
        else ''
    )
    msg = (
        'Please continue working on the task on whatever approach you think is suitable.\n'
        'If you think you have solved the task, please first send your answer to user through message and then finish the interaction.\n'
        f'{encaps_str}'
        'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n'
    )

    if state.history:
        # check if the last action has an answer, if so, early exit
        if try_parse is not None:
            last_action = next(
                (
                    event
                    for event in reversed(state.history)
                    if isinstance(event, Action)
                ),
                None,
            )
            ans = try_parse(last_action)
            if ans is not None:
                return '/exit'

        # check if the agent has tried to talk to the user 3 times, if so, let the agent know it can give up
        user_msgs = [
            event
            for event in state.history
            if isinstance(event, MessageAction) and event.source == 'user'
        ]
        if len(user_msgs) >= 2:
            # let the agent know that it can give up when it has tried 3 times
            return (
                msg
                + 'If you want to give up, run: <execute_bash> exit </execute_bash>.\n'
            )
    return msg


def cleanup() -> None:
    logger.info('Cleaning up child processes...')
    for process in mp.active_children():
        logger.info(f'Terminating child process: {process.name}')
        process.terminate()
        process.join()


def reset_logger_for_multiprocessing(
    logger: logging.Logger, instance_id: str, log_dir: str
) -> None:
    """Reset the logger for multiprocessing.

    Save logs to a separate file for each process, instead of trying to write to the
    same file/console from multiple processes.
    """
    # Set up logger
    log_file = os.path.join(
        log_dir,
        f'instance_{instance_id}.log',
    )
    # Remove all existing handlers from logger
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # add back the console handler to print ONE line
    logger.addHandler(get_console_handler())
    logger.info(
        f'Starting resolver for instance {instance_id}.\n'
        f'Hint: run "tail -f {log_file}" to see live logs in a separate shell'
    )
    # Remove all existing handlers from logger
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    logger.addHandler(file_handler)


def extract_image_urls(issue_body: str) -> list[str]:
    # Regular expression to match Markdown image syntax ![alt text](image_url)
    image_pattern = r'!\[.*?\]\((https?://[^\s)]+)\)'
    return re.findall(image_pattern, issue_body)


def extract_issue_references(body: str) -> list[int]:
    # First, remove code blocks as they may contain false positives
    body = re.sub(r'```.*?```', '', body, flags=re.DOTALL)

    # Remove inline code
    body = re.sub(r'`[^`]*`', '', body)

    # Remove URLs that contain hash symbols
    body = re.sub(r'https?://[^\s)]*#\d+[^\s)]*', '', body)

    # Now extract issue numbers, making sure they're not part of other text
    # The pattern matches #number that:
    # 1. Is at the start of text or after whitespace/punctuation
    # 2. Is followed by whitespace, punctuation, or end of text
    # 3. Is not part of a URL
    pattern = r'(?:^|[\s\[({]|[^\w#])#(\d+)(?=[\s,.\])}]|$)'
    return [int(match) for match in re.findall(pattern, body)]


def get_unique_uid(start_uid: int = 1000) -> int:
    existing_uids = set()
    with open('/etc/passwd', 'r') as passwd_file:
        for line in passwd_file:
            parts = line.split(':')
            if len(parts) > 2:
                try:
                    existing_uids.add(int(parts[2]))
                except ValueError:
                    continue

    while start_uid in existing_uids:
        start_uid += 1

    return start_uid
