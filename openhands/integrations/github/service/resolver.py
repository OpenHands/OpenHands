from datetime import datetime
from typing import Any

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.github.queries import (
    get_review_threads_graphql_query,
    get_thread_comments_graphql_query,
    get_thread_from_comment_graphql_query,
)
from openhands.integrations.github.service.base import GitHubMixinBase
from openhands.integrations.service_types import Comment


class GitHubResolverMixin(GitHubMixinBase):
    """Helper methods used for the GitHub Resolver
    """

    async def get_issue_or_pr_title_and_body(
        self, repository: str, issue_number: int
    ) -> tuple[str, str]:
        """Get the title and body of an issue.

        Args:
            repository: Repository name in format 'owner/repo'
            issue_number: The issue number

        Returns:
            A tuple of (title, body)
        """
        url = f'{self.BASE_URL}/repos/{repository}/issues/{issue_number}'
        response, _ = await self._make_request(url)
        title = response.get('title') or ''
        body = response.get('body') or ''
        return title, body

    async def get_issue_or_pr_comments(
        self, repository: str, issue_number: int, max_comments: int = 10
    ) -> list[Comment]:
        """Get comments for an issue.

        Args:
            repository: Repository name in format 'owner/repo'
            issue_number: The issue number
            discussion_id: Not used for GitHub (kept for compatibility with GitLab)

        Returns:
            List of Comment objects ordered by creation date
        """
        url = f'{self.BASE_URL}/repos/{repository}/issues/{issue_number}/comments'
        page = 1
        all_comments: list[dict] = []

        while len(all_comments) < max_comments:
            params = {
                'per_page': 10,
                'sort': 'created',
                'direction': 'asc',
                'page': page,
            }
            response, headers = await self._make_request(url, params=params)
            all_comments.extend(response or [])

            # Parse the Link header for rel="next"
            link_header = headers.get('Link', '')
            if 'rel="next"' not in link_header:
                break

            page += 1

        return self._process_raw_comments(all_comments)

    async def get_review_thread_comments(
        self,
        comment_id: str,
        repository: str,
        pr_number: int,
    ) -> list[Comment]:
        """Get all comments in a review thread starting from a specific comment.

        Uses GraphQL to traverse the reply chain from the given comment up to the root
        comment, then finds the review thread and returns all comments in the thread.

        Args:
            comment_id: The GraphQL node ID of any comment in the thread
            repo: Repository name
            pr_number: Pull request number

        Returns:
            List of Comment objects representing the entire thread
        """
        # Step 1: Use existing GraphQL query to get the comment and check for replyTo
        variables = {'commentId': comment_id}
        data = await self.execute_graphql_query(
            get_thread_from_comment_graphql_query, variables
        )

        comment_node = data.get('data', {}).get('node')
        if not comment_node:
            return []

        # Step 2: If replyTo exists, traverse to the root comment
        root_comment_id = comment_id
        reply_to = comment_node.get('replyTo')
        if reply_to:
            root_comment_id = reply_to['id']

        # Step 3: Get all review threads and find the one containing our root comment
        owner, repo = repository.split('/')
        thread_id = None
        after_cursor = None
        has_next_page = True

        while has_next_page and not thread_id:
            threads_variables: dict[str, Any] = {
                'owner': owner,
                'repo': repo,
                'number': pr_number,
                'first': 50,
            }
            if after_cursor:
                threads_variables['after'] = after_cursor  # type: ignore[unreachable]

            threads_data = await self.execute_graphql_query(
                get_review_threads_graphql_query, threads_variables
            )

            review_threads_data = (
                threads_data.get('data', {})
                .get('repository', {})
                .get('pullRequest', {})
                .get('reviewThreads', {})
            )

            review_threads = review_threads_data.get('nodes', [])
            page_info = review_threads_data.get('pageInfo', {})

            # Search for the thread containing our root comment
            for thread in review_threads:
                first_comments = thread.get('comments', {}).get('nodes', [])
                for first_comment in first_comments:
                    if first_comment.get('id') == root_comment_id:
                        thread_id = thread.get('id')
                        break
                if thread_id:
                    break

            # Update pagination variables
            has_next_page = page_info.get('hasNextPage', False)
            after_cursor = page_info.get('endCursor')

        if not thread_id:
            # Fallback: return just the comments we found during traversal
            logger.warning(
                f'Could not find review thread for comment {comment_id}, returning traversed comments'
            )
            return []

        # Step 4: Get all comments from the review thread using the thread ID
        all_thread_comments = []
        after_cursor = None
        has_next_page = True

        while has_next_page:
            comments_variables: dict[str, Any] = {}
            comments_variables['threadId'] = thread_id
            comments_variables['page'] = 50
            if after_cursor:
                comments_variables['after'] = after_cursor  # type: ignore[unreachable]

            thread_comments_data = await self.execute_graphql_query(
                get_thread_comments_graphql_query, comments_variables
            )

            thread_node = thread_comments_data.get('data', {}).get('node')
            if not thread_node:
                break

            comments_data = thread_node.get('comments', {})
            comments_nodes = comments_data.get('nodes', [])
            page_info = comments_data.get('pageInfo', {})

            all_thread_comments.extend(comments_nodes)

            has_next_page = page_info.get('hasNextPage', False)
            after_cursor = page_info.get('endCursor')

        return self._process_raw_comments(all_thread_comments)

    def _process_raw_comments(
        self, comments_data: list, max_comments: int = 10
    ) -> list[Comment]:
        """Convert raw comment data to Comment objects."""
        comments: list[Comment] = []
        for comment in comments_data:
            author = 'unknown'

            if comment.get('author'):
                author = comment.get('author', {}).get('login', 'unknown')
            elif comment.get('user'):
                author = comment.get('user', {}).get('login', 'unknown')

            comments.append(
                Comment(
                    id=str(comment.get('id', 'unknown')),
                    body=self._truncate_comment(comment.get('body', '')),
                    author=author,
                    created_at=datetime.fromisoformat(
                        comment.get('createdAt', '').replace('Z', '+00:00')
                    )
                    if comment.get('createdAt')
                    else datetime.fromtimestamp(0),
                    updated_at=datetime.fromisoformat(
                        comment.get('updatedAt', '').replace('Z', '+00:00')
                    )
                    if comment.get('updatedAt')
                    else datetime.fromtimestamp(0),
                    system=False,
                )
            )

        # Sort comments by creation date to maintain chronological order
        comments.sort(key=lambda c: c.created_at)
        return comments[-max_comments:]
