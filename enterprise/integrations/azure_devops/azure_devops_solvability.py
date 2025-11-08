"""Azure DevOps solvability analysis for work items and PRs."""

import asyncio
import time
from typing import TYPE_CHECKING

import requests
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

if TYPE_CHECKING:
    from integrations.azure_devops.azure_devops_view import AzureDevOpsViewType


def fetch_azure_devops_context(
    azure_devops_view: 'AzureDevOpsViewType',
    access_token: str,
) -> str:
    """Fetch full Azure DevOps work item or PR context.

    Includes title, description, comments, and labels/tags.

    Args:
        azure_devops_view: Azure DevOps view object
        access_token: Azure DevOps Personal Access Token or OAuth token

    Returns:
        A comprehensive string containing the work item/PR context
    """
    context_parts = []

    # Add title and description
    context_parts.append(f'Title: {azure_devops_view.title}')
    if azure_devops_view.description:
        context_parts.append(f'Description:\n{azure_devops_view.description}')

    # Fetch additional context from Azure DevOps API
    try:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}',
        }

        # Determine if this is a work item or PR
        if hasattr(azure_devops_view, 'work_item_id'):
            # Fetch work item details
            work_item_id = azure_devops_view.work_item_id
            organization = azure_devops_view.organization
            project = azure_devops_view.project

            url = f'https://dev.azure.com/{organization}/{project}/_apis/wit/workitems/{work_item_id}?$expand=all&api-version=7.2'
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                work_item = response.json()
                fields = work_item.get('fields', {})

                # Add tags if present
                tags = fields.get('System.Tags', '')
                if tags:
                    context_parts.append(f'Tags: {tags}')

                # Add state
                state = fields.get('System.State', '')
                if state:
                    context_parts.append(f'State: {state}')

                # Add work item type
                work_item_type = fields.get('System.WorkItemType', '')
                if work_item_type:
                    context_parts.append(f'Type: {work_item_type}')

        elif hasattr(azure_devops_view, 'pull_request_id'):
            # Fetch PR details
            pr_id = azure_devops_view.pull_request_id
            organization = azure_devops_view.organization
            project = azure_devops_view.project
            repository_id = azure_devops_view.repository_id

            url = f'https://dev.azure.com/{organization}/{project}/_apis/git/repositories/{repository_id}/pullRequests/{pr_id}?api-version=7.2'
            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code == 200:
                pr = response.json()

                # Add PR status
                status = pr.get('status', '')
                if status:
                    context_parts.append(f'Status: {status}')

                # Add labels/tags if present
                labels = pr.get('labels', [])
                if labels:
                    label_names = [label.get('name', '') for label in labels]
                    context_parts.append(f"Labels: {', '.join(label_names)}")

    except Exception as e:
        logger.warning(f'[Azure DevOps Solvability]: Error fetching context: {e}')

    # Add previous comments
    if hasattr(azure_devops_view, 'previous_comments'):
        for comment in azure_devops_view.previous_comments:
            author = getattr(comment, 'author', 'Unknown')
            body = getattr(comment, 'body', '')
            context_parts.append(f'- {author}: {body}')

    return '\n\n'.join(context_parts)


async def summarize_azure_devops_solvability(
    azure_devops_view: 'AzureDevOpsViewType',
    access_token: str,
    timeout: float = 60.0 * 5,
) -> str:
    """Generate a solvability summary for an Azure DevOps work item or PR.

    Args:
        azure_devops_view: Azure DevOps view object
        access_token: Azure DevOps access token for API access
        timeout: Maximum time in seconds to wait for the result (default: 300)

    Returns:
        The solvability summary as a markdown string

    Raises:
        ValueError: If LLM settings cannot be found for the user or feature is disabled
        asyncio.TimeoutError: If the operation exceeds the specified timeout
    """
    if not ENABLE_SOLVABILITY_ANALYSIS:
        raise ValueError('Solvability report feature is disabled')

    if azure_devops_view.user_info.keycloak_user_id is None:
        raise ValueError(
            f'[Solvability] No user ID found for user {azure_devops_view.user_info.username}'
        )

    # Load user's settings to get their LLM configuration
    store = SaasSettingsStore(
        user_id=azure_devops_view.user_info.keycloak_user_id,
        session_maker=session_maker,
        config=get_config(),
    )

    user_settings = await store.load()

    if user_settings is None:
        raise ValueError(
            f'[Solvability] No user settings found for user ID {azure_devops_view.user_info.user_id}'
        )

    # Check if solvability analysis is enabled for this user
    if not getattr(user_settings, 'enable_solvability_analysis', False):
        raise ValueError(
            f'Solvability analysis disabled for user {azure_devops_view.user_info.user_id}'
        )

    try:
        llm_config = LLMConfig(
            model=user_settings.llm_model,
            api_key=user_settings.llm_api_key.get_secret_value(),
            base_url=user_settings.llm_base_url,
        )
    except ValidationError as e:
        raise ValueError(
            f'[Solvability] Invalid LLM configuration for user {azure_devops_view.user_info.user_id}: {str(e)}'
        )

    # Fetch the full Azure DevOps context using the REST API
    start_time = time.time()
    context = fetch_azure_devops_context(azure_devops_view, access_token)
    logger.info(
        f'[Solvability] Grabbed Azure DevOps context for {azure_devops_view.conversation_id}',
        extra={
            'conversation_id': azure_devops_view.conversation_id,
            'response_latency': time.time() - start_time,
            'organization': getattr(azure_devops_view, 'organization', ''),
            'project': getattr(azure_devops_view, 'project', ''),
        },
    )

    # For comment-based triggers, include the triggering comment
    if hasattr(azure_devops_view, 'comment_body'):
        context += f'\n\nTriggering Comment:\n{azure_devops_view.comment_body}'

    # Load the solvability classifier
    solvability_classifier = load_classifier('default-classifier')

    async with asyncio.timeout(timeout):
        # Generate solvability report
        solvability_report: SolvabilityReport = await call_sync_from_async(
            lambda: solvability_classifier.solvability_report(
                context, llm_config=llm_config
            )
        )

        logger.info(
            f'[Solvability] Generated report for {azure_devops_view.conversation_id}',
            extra={
                'conversation_id': azure_devops_view.conversation_id,
                'report': solvability_report.model_dump(exclude=['issue']),
            },
        )

        # Create LLM registry and conversation stats
        llm_registry, conversation_stats, _ = create_registry_and_conversation_stats(
            get_config(),
            azure_devops_view.conversation_id,
            azure_devops_view.user_info.keycloak_user_id,
            None,
        )

        # Generate solvability summary from report
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
            f'[Solvability] Generated summary for {azure_devops_view.conversation_id}',
            extra={
                'conversation_id': azure_devops_view.conversation_id,
                'summary': solvability_summary.model_dump(exclude=['content']),
            },
        )

        return solvability_summary.format_as_markdown()
