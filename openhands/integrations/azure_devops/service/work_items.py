"""Work item operations for Azure DevOps integration."""

import re
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

    def _convert_markdown_links_to_html(self, text: str) -> str:
        """Convert Markdown to HTML for Azure DevOps comments.

        Azure DevOps work item comments support HTML but not Markdown.
        This function converts common Markdown patterns to HTML.

        Args:
            text: Text containing Markdown formatting

        Returns:
            Text with Markdown converted to HTML
        """
        if not text:
            return text

        # Convert headers (### Header -> <h3>Header</h3>)
        text = re.sub(r'^#### (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
        text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
        text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)

        # Convert bold (**text** or __text__ -> <strong>text</strong>)
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)

        # Convert inline code (`code` -> <code>code</code>)
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)

        # Convert Markdown links [text](url) -> <a href="url">text</a>
        text = re.sub(r'\[([^\]]+)\]\(([^\)]+)\)', r'<a href="\2">\1</a>', text)

        # Convert unordered lists (- item or * item -> <ul><li>item</li></ul>)
        lines = text.split('\n')
        result_lines = []
        in_list = False

        for line in lines:
            # Check if this is a list item (handle leading whitespace)
            list_match = re.match(r'^(\s*)[-*]\s+(.+)$', line)

            if list_match:
                if not in_list:
                    result_lines.append('<ul>')
                    in_list = True
                # Add the list item content (without the bullet)
                result_lines.append(f'<li>{list_match.group(2)}</li>')
            else:
                # Not a list item
                if in_list:
                    # Close the list
                    result_lines.append('</ul>')
                    in_list = False
                # Add the line as-is
                if line.strip():  # Only add non-empty lines
                    result_lines.append(line)
                else:
                    # Preserve empty lines for paragraph breaks
                    result_lines.append('')

        # Close list if still open at end
        if in_list:
            result_lines.append('</ul>')

        text = '\n'.join(result_lines)

        # Convert paragraph breaks (double newlines) to <br><br>
        text = re.sub(r'\n\s*\n', '<br><br>', text)
        # Convert remaining single newlines to <br>
        text = text.replace('\n', '<br>')

        return text

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

        # Convert Markdown links to HTML for better Azure DevOps compatibility
        comment_text_html = self._convert_markdown_links_to_html(comment_text)

        url = f'{self.base_url}/{org_enc}/{project_enc}/_apis/wit/workItems/{work_item_id}/comments?api-version=7.1-preview.4'

        payload = {
            'text': comment_text_html,
        }

        response, _ = await self._make_request(
            url=url, params=payload, method=RequestMethod.POST
        )

        logger.info(f'Added comment to work item {work_item_id} in project {project}')
        return response

    async def get_work_item(self, repository: str, work_item_id: int) -> dict:
        """Get full work item details including all fields.

        API Reference: https://learn.microsoft.com/en-us/rest/api/azure/devops/wit/work-items/get-work-item

        Args:
            repository: Repository name in format "organization/project/repo" (project extracted)
            work_item_id: The work item ID

        Returns:
            API response with complete work item information including all fields

        Raises:
            HTTPException: If the API request fails
        """
        org, project, _ = self._parse_repository(repository)

        # URL-encode components to handle spaces and special characters
        org_enc = self._encode_url_component(org)
        project_enc = self._encode_url_component(project)

        url = f'{self.base_url}/{org_enc}/{project_enc}/_apis/wit/workItems/{work_item_id}?api-version=7.1'

        response, _ = await self._make_request(url)

        logger.info(f'Fetched work item {work_item_id} from project {project}')
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
