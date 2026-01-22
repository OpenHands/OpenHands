from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader
from server.constants import WEB_HOST
from storage.org_store import OrgStore

from openhands.core.logger import openhands_logger as logger
from openhands.core.schema.agent import AgentState
from openhands.events import Event, EventSource
from openhands.events.action import (
    AgentFinishAction,
    MessageAction,
)
from openhands.events.event_filter import EventFilter
from openhands.events.event_store_abc import EventStoreABC
from openhands.events.observation.agent import AgentStateChangedObservation
from openhands.integrations.service_types import Repository
from openhands.storage.data_models.conversation_status import ConversationStatus
from openhands.utils.async_utils import call_sync_from_async

if TYPE_CHECKING:
    from openhands.server.conversation_manager.conversation_manager import (
        ConversationManager,
    )

# ---- DO NOT REMOVE ----
# WARNING: Langfuse depends on the WEB_HOST environment variable being set to track events.
HOST = WEB_HOST
# ---- DO NOT REMOVE ----

HOST_URL = f'https://{HOST}' if 'localhost' not in HOST else f'http://{HOST}'
GITHUB_WEBHOOK_URL = f'{HOST_URL}/integration/github/events'
GITLAB_WEBHOOK_URL = f'{HOST_URL}/integration/gitlab/events'
conversation_prefix = 'conversations/{}'
CONVERSATION_URL = f'{HOST_URL}/{conversation_prefix}'

# Toggle for auto-response feature that proactively starts conversations with users when workflow tests fail
ENABLE_PROACTIVE_CONVERSATION_STARTERS = (
    os.getenv('ENABLE_PROACTIVE_CONVERSATION_STARTERS', 'false').lower() == 'true'
)


def get_session_expired_message(username: str | None = None) -> str:
    """Get a user-friendly session expired message.

    Used by integrations to notify users when their Keycloak offline session
    has expired.

    Args:
        username: Optional username to mention in the message. If provided,
                  the message will include @username prefix (used by Git providers
                  like GitHub, GitLab, Slack). If None, returns a generic message
                  (used by Jira, Jira DC, Linear).

    Returns:
        A formatted session expired message
    """
    if username:
        return f'@{username} your session has expired. Please login again at [OpenHands Cloud]({HOST_URL}) and try again.'
    return f'Your session has expired. Please login again at [OpenHands Cloud]({HOST_URL}) and try again.'


# Toggle for solvability report feature
ENABLE_SOLVABILITY_ANALYSIS = (
    os.getenv('ENABLE_SOLVABILITY_ANALYSIS', 'false').lower() == 'true'
)

# Toggle for V1 GitHub resolver feature
ENABLE_V1_GITHUB_RESOLVER = (
    os.getenv('ENABLE_V1_GITHUB_RESOLVER', 'false').lower() == 'true'
)

ENABLE_V1_SLACK_RESOLVER = (
    os.getenv('ENABLE_V1_SLACK_RESOLVER', 'false').lower() == 'true'
)

OPENHANDS_RESOLVER_TEMPLATES_DIR = (
    os.getenv('OPENHANDS_RESOLVER_TEMPLATES_DIR')
    or 'openhands/integrations/templates/resolver/'
)
jinja_env = Environment(loader=FileSystemLoader(OPENHANDS_RESOLVER_TEMPLATES_DIR))


def get_oh_labels(web_host: str) -> tuple[str, str]:
    """Get the OpenHands labels based on the web host.

    Args:
        web_host: The web host string to check

    Returns:
        A tuple of (oh_label, inline_oh_label) where:
        - oh_label is 'openhands-exp' for staging/local hosts, 'openhands' otherwise
        - inline_oh_label is '@openhands-exp' for staging/local hosts, '@openhands' otherwise
    """
    web_host = web_host.strip()
    is_staging_or_local = 'staging' in web_host or 'local' in web_host
    oh_label = 'openhands-exp' if is_staging_or_local else 'openhands'
    inline_oh_label = '@openhands-exp' if is_staging_or_local else '@openhands'
    return oh_label, inline_oh_label


def get_summary_instruction():
    summary_instruction_template = jinja_env.get_template('summary_prompt.j2')
    summary_instruction = summary_instruction_template.render()
    return summary_instruction


async def get_user_v1_enabled_setting(user_id: str | None) -> bool:
    """Get the user's V1 conversation API setting.

    Args:
        user_id: The keycloak user ID

    Returns:
        True if V1 conversations are enabled for this user, False otherwise
    """
    if not user_id:
        return False

    org = await call_sync_from_async(
        OrgStore.get_current_org_from_keycloak_user_id, user_id
    )

    if not org or org.v1_enabled is None:
        return False

    return org.v1_enabled


def has_exact_mention(text: str, mention: str) -> bool:
    """Check if the text contains an exact mention (not part of a larger word).

    Args:
        text: The text to check for mentions
        mention: The mention to look for (e.g. "@openhands")

    Returns:
        bool: True if the exact mention is found, False otherwise

    Example:
        >>> has_exact_mention("Hello @openhands!", "@openhands")  # True
        >>> has_exact_mention("Hello @openhands-agent!", "@openhands")  # False
        >>> has_exact_mention("(@openhands)", "@openhands")  # True
        >>> has_exact_mention("user@openhands.com", "@openhands")  # False
        >>> has_exact_mention("Hello @OpenHands!", "@openhands")  # True (case-insensitive)
    """
    # Convert both text and mention to lowercase for case-insensitive matching
    text_lower = text.lower()
    mention_lower = mention.lower()

    pattern = re.escape(mention_lower)
    # Match mention that is not part of a larger word
    return bool(re.search(rf'(?:^|[^\w@]){pattern}(?![\w-])', text_lower))


def confirm_event_type(event: Event):
    return isinstance(event, AgentStateChangedObservation) and not (
        event.agent_state == AgentState.REJECTED
        or event.agent_state == AgentState.USER_CONFIRMED
        or event.agent_state == AgentState.USER_REJECTED
        or event.agent_state == AgentState.LOADING
        or event.agent_state == AgentState.RUNNING
    )


def get_readable_error_reason(reason: str):
    if reason == 'STATUS$ERROR_LLM_AUTHENTICATION':
        reason = 'Authentication with the LLM provider failed. Please check your API key or credentials'
    elif reason == 'STATUS$ERROR_LLM_SERVICE_UNAVAILABLE':
        reason = 'The LLM service is temporarily unavailable. Please try again later'
    elif reason == 'STATUS$ERROR_LLM_INTERNAL_SERVER_ERROR':
        reason = 'The LLM provider encountered an internal error. Please try again soon'
    elif reason == 'STATUS$ERROR_LLM_OUT_OF_CREDITS':
        reason = "You've run out of credits. Please top up to continue"
    elif reason == 'STATUS$ERROR_LLM_CONTENT_POLICY_VIOLATION':
        reason = 'Content policy violation. The output was blocked by content filtering policy'
    return reason


def get_summary_for_agent_state(
    observations: list[AgentStateChangedObservation], conversation_link: str
) -> str:
    unknown_error_msg = f'OpenHands encountered an unknown error. [See the conversation]({conversation_link}) for more information, or try again'

    if len(observations) == 0:
        logger.error(
            'Unknown error: No agent state observations found',
            extra={'conversation_link': conversation_link},
        )
        return unknown_error_msg

    observation: AgentStateChangedObservation = observations[0]
    state = observation.agent_state

    if state == AgentState.RATE_LIMITED:
        logger.warning(
            'Agent was rate limited',
            extra={
                'agent_state': state.value,
                'conversation_link': conversation_link,
                'observation_reason': getattr(observation, 'reason', None),
            },
        )
        return 'OpenHands was rate limited by the LLM provider. Please try again later.'

    if state == AgentState.ERROR:
        reason = observation.reason
        reason = get_readable_error_reason(reason)

        logger.error(
            'Agent encountered an error',
            extra={
                'agent_state': state.value,
                'conversation_link': conversation_link,
                'observation_reason': observation.reason,
                'readable_reason': reason,
            },
        )

        return f'OpenHands encountered an error: **{reason}**.\n\n[See the conversation]({conversation_link}) for more information.'

    if state == AgentState.AWAITING_USER_INPUT:
        logger.info(
            'Agent is awaiting user input',
            extra={
                'agent_state': state.value,
                'conversation_link': conversation_link,
                'observation_reason': getattr(observation, 'reason', None),
            },
        )
        return f'OpenHands is waiting for your input. [Continue the conversation]({conversation_link}) to provide additional instructions.'

    # Log unknown agent state as error
    logger.error(
        'Unknown error: Unhandled agent state',
        extra={
            'agent_state': state.value if hasattr(state, 'value') else str(state),
            'conversation_link': conversation_link,
            'observation_reason': getattr(observation, 'reason', None),
        },
    )
    return unknown_error_msg


def get_final_agent_observation(
    event_store: EventStoreABC,
) -> list[AgentStateChangedObservation]:
    events = list(
        event_store.search_events(
            filter=EventFilter(
                source=EventSource.ENVIRONMENT,
                include_types=(AgentStateChangedObservation,),
            ),
            limit=1,
            reverse=True,
        )
    )
    result = [e for e in events if isinstance(e, AgentStateChangedObservation)]
    assert len(result) == len(events)
    return result


def get_last_user_msg(event_store: EventStoreABC) -> list[MessageAction]:
    events = list(
        event_store.search_events(
            filter=EventFilter(
                source=EventSource.USER,
                include_types=(MessageAction,),
            ),
            limit=1,
            reverse=True,
        )
    )
    result = [e for e in events if isinstance(e, MessageAction)]
    assert len(result) == len(events)
    return result


def extract_summary_from_event_store(
    event_store: EventStoreABC, conversation_id: str
) -> str:
    """
    Get agent summary or alternative message depending on current AgentState
    """
    conversation_link = CONVERSATION_URL.format(conversation_id)
    summary_instruction = get_summary_instruction()

    instruction_events = list(
        event_store.search_events(
            filter=EventFilter(
                query=json.dumps(summary_instruction),
                source=EventSource.USER,
                include_types=(MessageAction,),
            ),
            limit=1,
            reverse=True,
        )
    )

    final_agent_observation = get_final_agent_observation(event_store)

    # Find summary instruction event ID
    if not instruction_events:
        logger.warning(
            'no_instruction_event_found', extra={'conversation_id': conversation_id}
        )
        return get_summary_for_agent_state(
            final_agent_observation, conversation_link
        )  # Agent did not receive summary instruction

    summary_events = list(
        event_store.search_events(
            filter=EventFilter(
                source=EventSource.AGENT,
                include_types=(MessageAction, AgentFinishAction),
            ),
            limit=1,
            reverse=True,
            start_id=instruction_events[0].id,
        )
    )

    if not summary_events:
        logger.warning(
            'no_agent_messages_found', extra={'conversation_id': conversation_id}
        )
        return get_summary_for_agent_state(
            final_agent_observation, conversation_link
        )  # Agent failed to generate summary

    summary_event = summary_events[0]
    if isinstance(summary_event, MessageAction):
        return summary_event.content

    assert isinstance(summary_event, AgentFinishAction)
    return summary_event.final_thought


async def get_event_store_from_conversation_manager(
    conversation_manager: ConversationManager, conversation_id: str
) -> EventStoreABC:
    agent_loop_infos = await conversation_manager.get_agent_loop_info(
        filter_to_sids={conversation_id}
    )
    if not agent_loop_infos or agent_loop_infos[0].status != ConversationStatus.RUNNING:
        raise RuntimeError(f'conversation_not_running:{conversation_id}')
    event_store = agent_loop_infos[0].event_store
    if not event_store:
        raise RuntimeError(f'event_store_missing:{conversation_id}')
    return event_store


async def get_last_user_msg_from_conversation_manager(
    conversation_manager: ConversationManager, conversation_id: str
):
    event_store = await get_event_store_from_conversation_manager(
        conversation_manager, conversation_id
    )
    return get_last_user_msg(event_store)


async def extract_summary_from_conversation_manager(
    conversation_manager: ConversationManager, conversation_id: str
) -> str:
    """
    Get agent summary or alternative message depending on current AgentState
    """

    event_store = await get_event_store_from_conversation_manager(
        conversation_manager, conversation_id
    )
    summary = extract_summary_from_event_store(event_store, conversation_id)
    return append_conversation_footer(summary, conversation_id)


def append_conversation_footer(message: str, conversation_id: str) -> str:
    """
    Append a small footer with the conversation URL to a message.

    Args:
        message: The original message content
        conversation_id: The conversation ID to link to

    Returns:
        The message with the conversation footer appended
    """
    conversation_link = CONVERSATION_URL.format(conversation_id)
    footer = f'\n\n[View full conversation]({conversation_link})'
    return message + footer


def infer_repo_from_message(user_msg: str) -> list[str]:
    """
    Extract all repository names in the format 'owner/repo' from various Git provider URLs
    and direct mentions in text. Supports GitHub, GitLab, and BitBucket.
    """
    normalized_msg = re.sub(r'\s+', ' ', user_msg.strip())

    git_url_pattern = (
        r'https?://(?:github\.com|gitlab\.com|bitbucket\.org)/'
        r'([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+?)(?:\.git)?'
        r'(?:[/?#].*?)?(?=\s|$|[^\w.-])'
    )

    # UPDATED: allow {{ owner/repo }} in addition to existing boundaries
    direct_pattern = (
        r'(?:^|\s|{{|[\[\(\'":`])'  # left boundary
        r'([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)'
        r'(?=\s|$|}}|[\]\)\'",.:`])'  # right boundary
    )

    matches: list[str] = []

    # Git URLs first (highest priority)
    for owner, repo in re.findall(git_url_pattern, normalized_msg):
        repo = re.sub(r'\.git$', '', repo)
        matches.append(f'{owner}/{repo}')

    # Direct mentions
    for owner, repo in re.findall(direct_pattern, normalized_msg):
        full_match = f'{owner}/{repo}'

        if (
            re.match(r'^\d+\.\d+/\d+\.\d+$', full_match)
            or re.match(r'^\d{1,2}/\d{1,2}$', full_match)
            or re.match(r'^[A-Z]/[A-Z]$', full_match)
            or repo.endswith(('.txt', '.md', '.py', '.js'))
            or ('.' in repo and len(repo.split('.')) > 2)
        ):
            continue

        if full_match not in matches:
            matches.append(full_match)

    return matches


def filter_potential_repos_by_user_msg(
    user_msg: str, user_repos: list[Repository]
) -> tuple[bool, list[Repository]]:
    """Filter repositories based on user message inference."""
    inferred_repos = infer_repo_from_message(user_msg)
    if not inferred_repos:
        return False, user_repos[0:99]

    final_repos = []
    for repo in user_repos:
        # Check if the repo matches any of the inferred repositories
        for inferred_repo in inferred_repos:
            if inferred_repo.lower() in repo.full_name.lower():
                final_repos.append(repo)
                break  # Avoid adding the same repo multiple times

    # no repos matched, return original list
    if len(final_repos) == 0:
        return False, user_repos[0:99]

    # Found exact match
    elif len(final_repos) == 1:
        return True, final_repos

    # Found partial matches
    return False, final_repos[0:99]


def markdown_to_jira_markup(markdown_text: str) -> str:
    """
    Convert markdown text to Jira Wiki Markup format.
    This function handles common markdown elements and converts them to their
    Jira Wiki Markup equivalents. It's designed to be exception-safe.
    Args:
        markdown_text: The markdown text to convert
    Returns:
        str: The converted Jira Wiki Markup text
    """
    if not markdown_text or not isinstance(markdown_text, str):
        return ''

    try:
        # Work with a copy to avoid modifying the original
        text = markdown_text

        # Convert headers (# ## ### #### ##### ######)
        text = re.sub(r'^#{6}\s+(.*?)$', r'h6. \1', text, flags=re.MULTILINE)
        text = re.sub(r'^#{5}\s+(.*?)$', r'h5. \1', text, flags=re.MULTILINE)
        text = re.sub(r'^#{4}\s+(.*?)$', r'h4. \1', text, flags=re.MULTILINE)
        text = re.sub(r'^#{3}\s+(.*?)$', r'h3. \1', text, flags=re.MULTILINE)
        text = re.sub(r'^#{2}\s+(.*?)$', r'h2. \1', text, flags=re.MULTILINE)
        text = re.sub(r'^#{1}\s+(.*?)$', r'h1. \1', text, flags=re.MULTILINE)

        # Convert code blocks first (before other formatting)
        text = re.sub(
            r'```(\w+)\n(.*?)\n```', r'{code:\1}\n\2\n{code}', text, flags=re.DOTALL
        )
        text = re.sub(r'```\n(.*?)\n```', r'{code}\n\1\n{code}', text, flags=re.DOTALL)

        # Convert inline code (`code`)
        text = re.sub(r'`([^`]+)`', r'{{\1}}', text)

        # Convert markdown formatting to Jira formatting
        # Use temporary placeholders to avoid conflicts between bold and italic conversion

        # First convert bold (double markers) to temporary placeholders
        text = re.sub(r'\*\*(.*?)\*\*', r'JIRA_BOLD_START\1JIRA_BOLD_END', text)
        text = re.sub(r'__(.*?)__', r'JIRA_BOLD_START\1JIRA_BOLD_END', text)

        # Now convert single asterisk italics
        text = re.sub(r'\*([^*]+?)\*', r'_\1_', text)

        # Convert underscore italics
        text = re.sub(r'(?<!_)_([^_]+?)_(?!_)', r'_\1_', text)

        # Finally, restore bold markers
        text = text.replace('JIRA_BOLD_START', '*')
        text = text.replace('JIRA_BOLD_END', '*')

        # Convert links [text](url)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'[\1|\2]', text)

        # Convert unordered lists (- or * or +)
        text = re.sub(r'^[\s]*[-*+]\s+(.*?)$', r'* \1', text, flags=re.MULTILINE)

        # Convert ordered lists (1. 2. etc.)
        text = re.sub(r'^[\s]*\d+\.\s+(.*?)$', r'# \1', text, flags=re.MULTILINE)

        # Convert strikethrough (~~text~~)
        text = re.sub(r'~~(.*?)~~', r'-\1-', text)

        # Convert horizontal rules (---, ***, ___)
        text = re.sub(r'^[\s]*[-*_]{3,}[\s]*$', r'----', text, flags=re.MULTILINE)

        # Convert blockquotes (> text)
        text = re.sub(r'^>\s+(.*?)$', r'bq. \1', text, flags=re.MULTILINE)

        # Convert tables (basic support)
        # This is a simplified table conversion - Jira tables are quite different
        lines = text.split('\n')
        in_table = False
        converted_lines = []

        for line in lines:
            if (
                '|' in line
                and line.strip().startswith('|')
                and line.strip().endswith('|')
            ):
                # Skip markdown table separator lines (contain ---)
                if '---' in line:
                    continue
                if not in_table:
                    in_table = True
                # Convert markdown table row to Jira table row
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                converted_line = '|' + '|'.join(cells) + '|'
                converted_lines.append(converted_line)
            elif in_table and line.strip() and '|' not in line:
                in_table = False
                converted_lines.append(line)
            else:
                in_table = False
                converted_lines.append(line)

        text = '\n'.join(converted_lines)

        return text

    except Exception as e:
        # Log the error but don't raise it - return original text as fallback
        print(f'Error converting markdown to Jira markup: {str(e)}')
        return markdown_text or ''
