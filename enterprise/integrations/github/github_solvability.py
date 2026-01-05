import asyncio
import time

from github import Auth, Github
from integrations.github.github_view import (
    GithubInlinePRComment,
    GithubIssueComment,
    GithubPRComment,
    GithubViewType,
)
from integrations.solvability.data import load_classifier
from integrations.solvability.models.report import SolvabilityReport
from integrations.solvability.models.summary import SolvabilitySummary
from integrations.utils import ENABLE_SOLVABILITY_ANALYSIS
from pydantic import ValidationError
from server.config import get_config
from storage.database import session_maker
from storage.saas_settings_store import SaasSettingsStore

from openhands.core.config import LLMConfig
from openhands.core.logger import openhands_logger as logger
from openhands.utils.async_utils import call_sync_from_async
from openhands.utils.utils import create_registry_and_conversation_stats


def fetch_github_issue_context(
    github_view: GithubViewType,
    user_token: str,
) -> str:
    """Fetch full GitHub issue/PR context including title, body, and comments.

    Args:
        full_repo_name: Full repository name in the format 'owner/repo'
        issue_number: The issue or PR number
        user_token: GitHub user access token
        max_comments: Maximum number of comments to fetch (default: 10)
        max_comment_length: Maximum length of each comment to include in the context (default: 500)

    Returns:
        A comprehensive string containing the issue/PR context
    """

    # Build context string
    context_parts = []

    # Add title and body
    context_parts.append(f'Title: {github_view.title}')
    context_parts.append(f'Description:\n{github_view.description}')

    with Github(auth=Auth.Token(user_token)) as github_client:
        repo = github_client.get_repo(github_view.full_repo_name)
        issue = repo.get_issue(github_view.issue_number)
        if issue.labels:
            labels = [label.name for label in issue.labels]
            context_parts.append(f"Labels: {', '.join(labels)}")

    for comment in github_view.previous_comments:
        context_parts.append(f'- {comment.author}: {comment.body}')

    return '\n\n'.join(context_parts)


async def summarize_issue_solvability(
    github_view: GithubViewType,
    user_token: str,
    timeout: float = 60.0 * 5,
) -> str:
    """Generate a solvability summary for an issue using the resolver view interface.

    Args:
        resolver_view: A resolver view interface instance (e.g., GithubIssue, GithubPRComment)
        user_token: GitHub user access token for API access
        timeout: Maximum time in seconds to wait for the result (default: 60.0)

    Returns:
        The solvability summary as a string

    Raises:
        ValueError: If LLM settings cannot be found for the user
        asyncio.TimeoutError: If the operation exceeds the specified timeout
    """
    if not ENABLE_SOLVABILITY_ANALYSIS:
        raise ValueError('Solvability report feature is disabled')

    if github_view.user_info.keycloak_user_id is None:
        raise ValueError(
            f'[Solvability] No user ID found for user {github_view.user_info.username}'
        )

    # Grab the user's information so we can load their LLM configuration
    store = SaasSettingsStore(
        user_id=github_view.user_info.keycloak_user_id,
        session_maker=session_maker,
        config=get_config(),
    )

    user_settings = await store.load()

    if user_settings is None:
        raise ValueError(
            f'[Solvability] No user settings found for user ID {github_view.user_info.user_id}'
        )

    # Check if solvability analysis is enabled for this user, exit early if
    # needed
    if not getattr(user_settings, 'enable_solvability_analysis', False):
        raise ValueError(
            f'Solvability analysis disabled for user {github_view.user_info.user_id}'
        )

    try:
        llm_config = LLMConfig(
            model=user_settings.llm_model,
            api_key=user_settings.llm_api_key.get_secret_value(),
            base_url=user_settings.llm_base_url,
        )
    except ValidationError as e:
        raise ValueError(
            f'[Solvability] Invalid LLM configuration for user {github_view.user_info.user_id}: {str(e)}'
        )

    # Fetch the full GitHub issue/PR context using the GitHub API
    start_time = time.time()
    issue_context = fetch_github_issue_context(github_view, user_token)
    logger.info(
        f'[Solvability] Grabbed issue context for {github_view.conversation_id}',
        extra={
            'conversation_id': github_view.conversation_id,
            'response_latency': time.time() - start_time,
            'full_repo_name': github_view.full_repo_name,
            'issue_number': github_view.issue_number,
        },
    )

    # For comment-based triggers, also include the specific comment that triggered the action
    if isinstance(
        github_view, (GithubIssueComment, GithubPRComment, GithubInlinePRComment)
    ):
        issue_context += f'\n\nTriggering Comment:\n{github_view.comment_body}'

    solvability_classifier = load_classifier('default-classifier')

    async with asyncio.timeout(timeout):
        solvability_report: SolvabilityReport = await call_sync_from_async(
            lambda: solvability_classifier.solvability_report(
                issue_context, llm_config=llm_config
            )
        )

        logger.info(
            f'[Solvability] Generated report for {github_view.conversation_id}',
            extra={
                'conversation_id': github_view.conversation_id,
                'report': solvability_report.model_dump(exclude=['issue']),
            },
        )

        llm_registry, conversation_stats, _ = create_registry_and_conversation_stats(
            get_config(),
            github_view.conversation_id,
            github_view.user_info.keycloak_user_id,
            None,
        )

        solvability_summary = await call_sync_from_async(
            lambda: SolvabilitySummary.from_report(
                solvability_report,
                llm=llm_registry.get_llm(
                    service_id='solvability_analysis', config=llm_config
                ),
            )
        )
        conversation_stats.save_metrics()

        logger.info(
            f'[Solvability] Generated summary for {github_view.conversation_id}',
            extra={
                'conversation_id': github_view.conversation_id,
                'summary': solvability_summary.model_dump(exclude=['content']),
            },
        )

        return solvability_summary.format_as_markdown()
