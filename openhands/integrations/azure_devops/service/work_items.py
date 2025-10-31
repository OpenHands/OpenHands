"""Work item operations for Azure DevOps integration."""

from datetime import datetime

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.azure_devops.service.base import AzureDevOpsMixinBase
from openhands.integrations.service_types import Comment, RequestMethod


class AzureDevOpsWorkItemsMixin(AzureDevOpsMixinBase):
    """Mixin for Azure DevOps work item operations.

    Work Items are unique to Azure DevOps and represent tasks, bugs, user stories, etc.
    in Azure Boards. This mixin provides methods to interact with work item comments.
    """

    def _truncate_comment(self, comment: str, max_length: int = 1000) -> str:
        """Truncate comment to max length."""
        if len(comment) <= max_length:
            return comment
        return comment[:max_length] + '...'

    async def add_work_item_comment(
        self, repository: str, work_item_id: int, comment_text: str
    ) -> dict:
        """Add a comment to an Azure DevOps work item.

        API Reference: https://learn.microsoft.com/en-us/rest/api/azure/devops/wit/comments/add-comment

        Args:
            repository: Repository name in format "organization/project/repo" (project extracted)
            work_item_id: The work item ID
            comment_text: The comment text to post

        Returns:
            API response with created comment information

        Raises:
            HTTPException: If the API request fails
        """
        org, project, _ = self._parse_repository(repository)

        # URL-encode components to handle spaces and special characters
        org_enc = self._encode_url_component(org)
        project_enc = self._encode_url_component(project)

        url = f'{self.base_url}/{org_enc}/{project_enc}/_apis/wit/workItems/{work_item_id}/comments?api-version=7.1-preview.4'

        payload = {
            'text': comment_text,
        }

        response, _ = await self._make_request(
            url=url, params=payload, method=RequestMethod.POST
        )

        logger.info(f'Added comment to work item {work_item_id} in project {project}')
        return response

    async def get_work_item_comments(
        self, repository: str, work_item_id: int, max_comments: int = 100
    ) -> list[Comment]:
        """Get all comments from a work item.

        API Reference: https://learn.microsoft.com/en-us/rest/api/azure/devops/wit/comments/get-comments

        Args:
            repository: Repository name in format "organization/project/repo" (project extracted)
            work_item_id: The work item ID
            max_comments: Maximum number of comments to return

        Returns:
            List of Comment objects sorted by creation date
        """
        org, project, _ = self._parse_repository(repository)

        # URL-encode components to handle spaces and special characters
        org_enc = self._encode_url_component(org)
        project_enc = self._encode_url_component(project)

        url = f'{self.base_url}/{org_enc}/{project_enc}/_apis/wit/workItems/{work_item_id}/comments?api-version=7.1-preview.4'

        response, _ = await self._make_request(url)

        comments_data = response.get('comments', [])
        all_comments: list[Comment] = []

        for comment_data in comments_data:
            # Extract author information
            author_info = comment_data.get('createdBy', {})
            author = author_info.get('displayName', 'unknown')

            # Parse dates
            created_at = (
                datetime.fromisoformat(
                    comment_data.get('createdDate', '').replace('Z', '+00:00')
                )
                if comment_data.get('createdDate')
                else datetime.fromtimestamp(0)
            )

            modified_at = (
                datetime.fromisoformat(
                    comment_data.get('modifiedDate', '').replace('Z', '+00:00')
                )
                if comment_data.get('modifiedDate')
                else created_at
            )

            comment = Comment(
                id=str(comment_data.get('id', 0)),
                body=self._truncate_comment(comment_data.get('text', '')),
                author=author,
                created_at=created_at,
                updated_at=modified_at,
                system=False,
            )

            all_comments.append(comment)

        # Sort by creation date and limit
        all_comments.sort(key=lambda c: c.created_at)
        return all_comments[:max_comments]

    async def add_work_item_reaction(
        self, repository: str, work_item_id: int, reaction_type: str = ':thumbsup:'
    ) -> dict:
        comment_text = f'{reaction_type} OpenHands is processing this work item...'
        return await self.add_work_item_comment(repository, work_item_id, comment_text)
