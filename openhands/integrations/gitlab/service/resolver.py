from datetime import datetime

from openhands.integrations.gitlab.service.base import GitLabMixinBase
from openhands.integrations.service_types import Comment


class GitLabResolverMixin(GitLabMixinBase):
    """Helper methods used for the GitLab Resolver
    """

    async def get_review_thread_comments(
        self, project_id: str, issue_iid: int, discussion_id: str
    ) -> list[Comment]:
        url = (
            f'{self.BASE_URL}/projects/{project_id}'
            f'/merge_requests/{issue_iid}/discussions/{discussion_id}'
        )

        # Single discussion fetch; notes are returned inline.
        response, _ = await self._make_request(url)
        notes = response.get('notes') or []
        return self._process_raw_comments(notes)

    async def get_issue_or_mr_title_and_body(
        self, project_id: str, issue_number: int, is_mr: bool = False
    ) -> tuple[str, str]:
        """Get the title and body of an issue or merge request.

        Args:
            repository: Repository name in format 'owner/repo' or 'domain/owner/repo'
            issue_number: The issue/MR IID within the project
            is_mr: If True, treat as merge request; if False, treat as issue;
                   if None, try issue first then merge request (default behavior)

        Returns:
            A tuple of (title, body)
        """
        if is_mr:
            url = f'{self.BASE_URL}/projects/{project_id}/merge_requests/{issue_number}'
            response, _ = await self._make_request(url)
            title = response.get('title') or ''
            body = response.get('description') or ''
            return title, body

        url = f'{self.BASE_URL}/projects/{project_id}/issues/{issue_number}'
        response, _ = await self._make_request(url)
        title = response.get('title') or ''
        body = response.get('description') or ''
        return title, body

    async def get_issue_or_mr_comments(
        self,
        project_id: str,
        issue_number: int,
        max_comments: int = 10,
        is_mr: bool = False,
    ) -> list[Comment]:
        """Get comments for an issue or merge request.

        Args:
            repository: Repository name in format 'owner/repo' or 'domain/owner/repo'
            issue_number: The issue/MR IID within the project
            max_comments: Maximum number of comments to retrieve
            is_pr: If True, treat as merge request; if False, treat as issue;
                   if None, try issue first then merge request (default behavior)

        Returns:
            List of Comment objects ordered by creation date
        """
        all_comments: list[Comment] = []
        page = 1
        per_page = min(max_comments, 10)

        url = (
            f'{self.BASE_URL}/projects/{project_id}/merge_requests/{issue_number}/discussions'
            if is_mr
            else f'{self.BASE_URL}/projects/{project_id}/issues/{issue_number}/notes'
        )

        while len(all_comments) < max_comments:
            params = {
                'per_page': per_page,
                'page': page,
                'order_by': 'created_at',
                'sort': 'asc',
            }

            response, headers = await self._make_request(url, params)
            if not response:
                break

            if is_mr:
                for discussions in response:
                    # Keep root level comments
                    all_comments.append(discussions['notes'][0])
            else:
                all_comments.extend(response)

            link_header = headers.get('Link', '')
            if 'rel="next"' not in link_header:
                break

            page += 1

        return self._process_raw_comments(all_comments)

    def _process_raw_comments(
        self, comments: list, max_comments: int = 10
    ) -> list[Comment]:
        """Helper method to fetch comments from a given URL with pagination."""
        all_comments: list[Comment] = []
        for comment_data in comments:
            comment = Comment(
                id=str(comment_data.get('id', 'unknown')),
                body=self._truncate_comment(comment_data.get('body', '')),
                author=comment_data.get('author', {}).get('username', 'unknown'),
                created_at=datetime.fromisoformat(
                    comment_data.get('created_at', '').replace('Z', '+00:00')
                )
                if comment_data.get('created_at')
                else datetime.fromtimestamp(0),
                updated_at=datetime.fromisoformat(
                    comment_data.get('updated_at', '').replace('Z', '+00:00')
                )
                if comment_data.get('updated_at')
                else datetime.fromtimestamp(0),
                system=comment_data.get('system', False),
            )
            all_comments.append(comment)

        # Sort comments by creation date and return the most recent ones
        all_comments.sort(key=lambda c: c.created_at)
        return all_comments[-max_comments:]
