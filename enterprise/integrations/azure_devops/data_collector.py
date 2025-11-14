"""Azure DevOps data collector for tracking resolver interactions."""

import json
import os
import re
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import requests
from integrations.models import Message
from integrations.types import PRStatus, ResolverViewInterface
from integrations.utils import HOST
from storage.openhands_pr import OpenhandsPR
from storage.openhands_pr_store import OpenhandsPRStore

from openhands.core.config import load_openhands_config
from openhands.core.logger import openhands_logger as logger
from openhands.integrations.service_types import ProviderType
from openhands.storage import get_file_store
from openhands.storage.locations import get_conversation_dir

config = load_openhands_config()
file_store = get_file_store(config.file_store, config.file_store_path)


COLLECT_AZURE_DEVOPS_INTERACTIONS = (
    os.getenv('COLLECT_AZURE_DEVOPS_INTERACTIONS', 'false') == 'true'
)


class TriggerType(str, Enum):
    """Types of triggers for Azure DevOps resolver interactions."""

    WORKITEM_COMMENT = 'workitem-comment'
    PR_COMMENT = 'pr-comment'


class AzureDevOpsDataCollector:
    """Saves data on Azure DevOps Resolver Interactions.

    1. We always save
        - Resolver trigger (work item comment or PR comment)
        - Metadata (who started the job, project name, work item/PR number)

    2. We save data for the type of interaction
        a. For work item comments, we save
            - {conversation_dir}/{conversation_id}/azure_devops_data/workitem__{project}_{work_item_id}.json
                - work item ID
                - trigger
                - metadata
                - body (description)
                - title
                - comments

        b. For PR comments, we save
            - {conversation_dir}/{conversation_id}/azure_devops_data/pr__{project}_{pr_number}.json
                - pr_number
                - metadata
                - body (description)
                - title
                - commits/authors

    3. For all PRs that were closed/merged with the resolver, we save
        - azure_devops_data/prs/{project}_{pr_number}/data.json
            - pr_number
            - title
            - body
            - commits/authors
            - code diffs
            - merge status (either merged/closed/abandoned)
    """

    def __init__(self, access_token: str | None = None):
        """Initialize Azure DevOps data collector.

        Args:
            access_token: Azure DevOps Personal Access Token or OAuth token (optional for basic tracking)
        """
        self.access_token = access_token
        self.file_store = file_store
        self.workitems_path = 'azure_devops_data/workitem-{}-{}/data.json'
        self.pr_path = 'azure_devops_data/pr-{}-{}/data.json'
        self.full_saved_pr_path = 'prs/azure_devops/{}-{}/data.json'
        self.conversation_id = None

    def _create_file_name(
        self,
        path: str,
        project_or_repo_id: str,
        number: int,
        conversation_id: str | None,
    ):
        """Create file path for saving data."""
        suffix = path.format(project_or_repo_id, number)

        if conversation_id:
            return f'{get_conversation_dir(conversation_id)}{suffix}'

        return suffix

    def _save_data(self, path: str, data: dict[str, Any]):
        """Save data to a path."""
        self.file_store.write(path, json.dumps(data))

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for Azure DevOps API requests."""
        return {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.access_token}',
        }

    def _check_openhands_author(self, name: str | None, login: str | None) -> bool:
        """Check if an author is an OpenHands bot account."""
        if not name and not login:
            return False

        name_lower = name.lower() if name else ''
        login_lower = login.lower() if login else ''

        return (
            name_lower == 'openhands'
            or login_lower == 'openhands'
            or login_lower == 'openhands-agent'
            or login_lower == 'openhands-ai'
            or login_lower == 'openhands-staging'
            or login_lower == 'openhands-exp'
            or 'openhands' in login_lower
        )

    def _get_work_item_comments(
        self,
        organization: str,
        project: str,
        work_item_id: int,
        conversation_id: str,
    ) -> list[dict[str, Any]]:
        """Retrieve all comments from a work item until a comment with conversation_id is found.

        Args:
            organization: Azure DevOps organization name
            project: Project name
            work_item_id: Work item ID
            conversation_id: Conversation ID to stop at

        Returns:
            List of comment dictionaries
        """
        try:
            url = f'https://dev.azure.com/{organization}/{project}/_apis/wit/workItems/{work_item_id}/comments?api-version=7.2'
            headers = self._get_headers()

            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code != 200:
                logger.warning(
                    f'Failed to fetch work item comments: {response.status_code}'
                )
                return []

            data = response.json()
            comments = []

            for comment in data.get('comments', []):
                comment_text = comment.get('text', '')

                # If we find a comment containing conversation_id, stop collecting
                if conversation_id in comment_text:
                    break

                comments.append(
                    {
                        'id': comment.get('id'),
                        'text': comment_text,
                        'created_date': comment.get('createdDate'),
                        'created_by': comment.get('createdBy', {}).get(
                            'displayName', ''
                        ),
                    }
                )

            return comments

        except Exception as e:
            logger.exception(f'Error fetching work item comments: {e}')
            return []

    def _get_pr_commits(
        self,
        organization: str,
        project: str,
        repository_id: str,
        pull_request_id: int,
    ) -> list[dict[str, Any]]:
        """Retrieve commits for a pull request.

        Args:
            organization: Azure DevOps organization name
            project: Project name
            repository_id: Repository ID
            pull_request_id: Pull request ID

        Returns:
            List of commit dictionaries
        """
        try:
            url = f'https://dev.azure.com/{organization}/{project}/_apis/git/repositories/{repository_id}/pullRequests/{pull_request_id}/commits?api-version=7.2'
            headers = self._get_headers()

            response = requests.get(url, headers=headers, timeout=30)

            if response.status_code != 200:
                logger.warning(f'Failed to fetch PR commits: {response.status_code}')
                return []

            data = response.json()
            commits = []

            for commit in data.get('value', []):
                author_info = commit.get('author', {})
                commits.append(
                    {
                        'commitId': commit.get('commitId'),
                        'comment': commit.get('comment', ''),
                        'author': {
                            'name': author_info.get('name'),
                            'email': author_info.get('email'),
                            'date': author_info.get('date'),
                        },
                    }
                )

            return commits

        except Exception as e:
            logger.exception(f'Error fetching PR commits: {e}')
            return []

    def _count_openhands_activity(
        self, commits: list[dict[str, Any]]
    ) -> tuple[int, int, int]:
        """Count OpenHands commits in Azure DevOps.

        Args:
            commits: List of commit dictionaries

        Returns:
            Tuple of (openhands_commit_count, 0, 0) - other counts not implemented yet
        """
        openhands_commit_count = 0

        for commit in commits:
            author = commit.get('author', {})
            author_name = author.get('name', '').lower() if author.get('name') else ''
            author_email = (
                author.get('email', '').lower() if author.get('email') else ''
            )

            if self._check_openhands_author(author_name, author_email):
                openhands_commit_count += 1

        return (openhands_commit_count, 0, 0)

    def _check_for_conversation_url(self, body: str) -> Optional[str]:
        """Extract conversation ID from body text."""
        conversation_pattern = re.search(
            rf'https://{HOST}/conversations/([a-zA-Z0-9-]+)(?:\s|[.,;!?)]|$)', body
        )
        if conversation_pattern:
            return conversation_pattern.group(1)

        return None

    def _is_pr_completed(self, payload: dict) -> bool:
        """Check if PR was completed (closed, abandoned, or merged).

        Args:
            payload: Raw webhook payload

        Returns:
            True if PR is completed
        """
        event_type = payload.get('eventType', '')
        if event_type != 'git.pullrequest.updated':
            return False

        resource = payload.get('resource', {})
        status = resource.get('status', '').lower()

        # Azure DevOps PR statuses: active, abandoned, completed
        return status in ('abandoned', 'completed')

    def _track_completed_pr(self, payload: dict):
        """Track PR completed event (merged, closed, or abandoned).

        Args:
            payload: Raw webhook payload from Azure DevOps
        """
        try:
            resource = payload.get('resource', {})
            repository = resource.get('repository', {})
            project = repository.get('project', {})

            repo_id = repository.get('id', '')
            repo_name = repository.get('name', '')
            pr_number = resource.get('pullRequestId', 0)
            project_name = project.get('name', '')
            status = resource.get('status', '').lower()

            # Map Azure DevOps status to our PRStatus enum
            if status == 'completed' and resource.get('mergeStatus') == 'succeeded':
                pr_status = PRStatus.MERGED
                merged = True
            elif status == 'abandoned':
                pr_status = PRStatus.CLOSED
                merged = False
            else:
                pr_status = PRStatus.CLOSED
                merged = False

            # Extract PR metadata
            _title = resource.get('title', '')
            _description = resource.get('description', '')
            _created_by = resource.get('createdBy', {})
            created_date = resource.get('creationDate')
            closed_date = resource.get('closedDate')

            # Parse dates
            created_at = created_date
            if closed_date:
                closed_at = datetime.fromisoformat(closed_date.replace('Z', '+00:00'))
            else:
                closed_at = datetime.now()

            # Note: Azure DevOps REST API doesn't provide these metrics directly in webhook
            # We would need to make additional API calls to get accurate counts
            num_commits = 0
            num_changed_files = 0
            num_additions = 0
            num_deletions = 0

            store = OpenhandsPRStore.get_instance()

            pr = OpenhandsPR(
                repo_name=f'{project_name}/{repo_name}',
                repo_id=repo_id,
                pr_number=pr_number,
                status=pr_status,
                provider=ProviderType.AZURE_DEVOPS.value,
                installation_id='',  # Azure DevOps doesn't have installation IDs
                private=True,  # Assume private for Azure DevOps
                num_reviewers=0,
                num_commits=num_commits,
                num_review_comments=0,
                num_changed_files=num_changed_files,
                num_additions=num_additions,
                num_deletions=num_deletions,
                merged=merged,
                created_at=created_at,
                closed_at=closed_at,
                openhands_helped_author=None,
                num_openhands_commits=None,
                num_openhands_review_comments=None,
                num_general_comments=0,
            )

            store.insert_pr(pr)
            logger.info(
                f'[Azure DevOps]: Tracked PR {pr_status}: {project_name}/{repo_name}#{pr_number}'
            )

        except Exception as e:
            logger.exception(f'[Azure DevOps]: Error tracking completed PR: {e}')

    async def save_full_pr(self, openhands_pr: OpenhandsPR) -> None:
        """Save comprehensive PR information using Azure DevOps REST API.

        Args:
            openhands_pr: OpenhandsPR object with PR metadata

        Saves:
        - Repo metadata (repo name, project)
        - PR metadata (number, title, body, author)
        - Commit information (sha, authors, message)
        - Merge status
        - Num openhands commits
        """
        try:
            pr_number = openhands_pr.pr_number
            _repo_id = openhands_pr.repo_id
            repo_name = openhands_pr.repo_name

            # Parse project and repo from repo_name (format: project/repo)
            parts = repo_name.split('/')
            if len(parts) != 2:
                logger.warning(f'[Azure DevOps]: Invalid repo_name format: {repo_name}')
                return

            _project_name = parts[0]
            _repository_name = parts[1]

            # Extract organization from somewhere (we'll need to pass this in)
            # For now, we'll skip this since we don't have the organization info

            # Get PR details
            # url = f'https://dev.azure.com/{organization}/{project_name}/_apis/git/repositories/{repo_id}/pullRequests/{pr_number}?api-version=7.2'

            # TODO: Implement full PR data collection when we have organization context
            logger.info(
                f'[Azure DevOps]: Full PR save for #{pr_number} in {repo_name} - not fully implemented yet'
            )

        except Exception as e:
            logger.exception(f'[Azure DevOps]: Error saving full PR: {e}')

    def process_payload(self, message: Message):
        """Process incoming Azure DevOps webhook payload.

        Args:
            message: Message object containing webhook payload
        """
        if not COLLECT_AZURE_DEVOPS_INTERACTIONS:
            return

        try:
            raw_payload = message.message if isinstance(message.message, dict) else {}

            if self._is_pr_completed(raw_payload):
                self._track_completed_pr(raw_payload)

        except Exception as e:
            logger.exception(f'[Azure DevOps]: Error processing payload: {e}')

    async def save_data(self, azure_devops_view: ResolverViewInterface):
        """Save data for Azure DevOps resolver interaction.

        Args:
            azure_devops_view: View object containing interaction data
        """
        if not COLLECT_AZURE_DEVOPS_INTERACTIONS:
            return

        # TODO: Implement when we have Azure DevOps view interface
        return
