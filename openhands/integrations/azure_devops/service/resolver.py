from datetime import datetime

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.azure_devops.service.base import AzureDevOpsMixinBase
from openhands.integrations.service_types import Comment


class AzureDevOpsResolverMixin(AzureDevOpsMixinBase):
    """Helper methods used for the Azure DevOps Resolver."""

    async def get_issue_or_pr_title_and_body(
        self, repository: str, issue_number: int
    ) -> tuple[str, str]:
        """Get the title and body of a pull request or work item.

        First attempts to get as a PR, then falls back to work item if not found.

        Args:
            repository: Repository name in format 'organization/project/repo'
            issue_number: The PR number or work item ID

        Returns:
            A tuple of (title, body)
        """
        org, project, repo = self._parse_repository(repository)

        # URL-encode components to handle spaces and special characters
        org_enc = self._encode_url_component(org)
        project_enc = self._encode_url_component(project)
        repo_enc = self._encode_url_component(repo)

        # Try to get as a pull request first
        try:
            pr_url = f'{self.base_url}/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/pullrequests/{issue_number}?api-version=7.1'
            response, _ = await self._make_request(pr_url)
            title = response.get('title') or ''
            body = response.get('description') or ''
            return title, body
        except Exception as pr_error:
            logger.debug(f'Failed to get as PR: {pr_error}, trying as work item')

        # Fall back to work item
        try:
            wi_url = f'{self.base_url}/{org_enc}/{project_enc}/_apis/wit/workitems/{issue_number}?api-version=7.1'
            response, _ = await self._make_request(wi_url)
            fields = response.get('fields', {})
            title = fields.get('System.Title') or ''
            body = fields.get('System.Description') or ''
            return title, body
        except Exception as wi_error:
            logger.error(f'Failed to get as work item: {wi_error}')
            return '', ''

    async def get_issue_or_pr_comments(
        self, repository: str, issue_number: int, max_comments: int = 10
    ) -> list[Comment]:
        """Get comments for a pull request or work item.

        First attempts to get PR comments, then falls back to work item comments if not found.

        Args:
            repository: Repository name in format 'organization/project/repo'
            issue_number: The PR number or work item ID
            max_comments: Maximum number of comments to return

        Returns:
            List of Comment objects ordered by creation date
        """
        # Try to get PR comments first
        try:
            comments = await self.get_pr_comments(  # type: ignore[attr-defined]
                repository, issue_number, max_comments
            )
            if comments:
                return comments
        except Exception as pr_error:
            logger.debug(f'Failed to get PR comments: {pr_error}, trying work item')

        # Fall back to work item comments
        try:
            return await self.get_work_item_comments(  # type: ignore[attr-defined]
                repository, issue_number, max_comments
            )
        except Exception as wi_error:
            logger.error(f'Failed to get work item comments: {wi_error}')
            return []

    async def get_review_thread_comments(
        self,
        thread_id: int,
        repository: str,
        pr_number: int,
        max_comments: int = 10,
    ) -> list[Comment]:
        """Get all comments in a specific PR review thread.

        Azure DevOps organizes PR comments into threads. This method retrieves
        all comments from a specific thread.

        Args:
            thread_id: The thread ID
            repository: Repository name in format 'organization/project/repo'
            pr_number: Pull request number
            max_comments: Maximum number of comments to return

        Returns:
            List of Comment objects representing the thread
        """
        org, project, repo = self._parse_repository(repository)

        # URL-encode components to handle spaces and special characters
        org_enc = self._encode_url_component(org)
        project_enc = self._encode_url_component(project)
        repo_enc = self._encode_url_component(repo)

        url = f'{self.base_url}/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/pullrequests/{pr_number}/threads/{thread_id}?api-version=7.1'

        try:
            response, _ = await self._make_request(url)
            comments_data = response.get('comments', [])

            all_comments: list[Comment] = []

            for comment_data in comments_data:
                # Extract author information
                author_info = comment_data.get('author', {})
                author = author_info.get('displayName', 'unknown')

                # Parse dates
                created_at = (
                    datetime.fromisoformat(
                        comment_data.get('publishedDate', '').replace('Z', '+00:00')
                    )
                    if comment_data.get('publishedDate')
                    else datetime.fromtimestamp(0)
                )

                updated_at = (
                    datetime.fromisoformat(
                        comment_data.get('lastUpdatedDate', '').replace('Z', '+00:00')
                    )
                    if comment_data.get('lastUpdatedDate')
                    else created_at
                )

                # Check if it's a system comment
                is_system = comment_data.get('commentType', 1) != 1  # 1 = text comment

                comment = Comment(
                    id=str(comment_data.get('id', 0)),
                    body=self._truncate_comment(comment_data.get('content', '')),
                    author=author,
                    created_at=created_at,
                    updated_at=updated_at,
                    system=is_system,
                )

                all_comments.append(comment)

            # Sort by creation date and limit
            all_comments.sort(key=lambda c: c.created_at)
            return all_comments[:max_comments]

        except Exception as error:
            logger.error(f'Failed to get thread {thread_id} comments: {error}')
            return []
