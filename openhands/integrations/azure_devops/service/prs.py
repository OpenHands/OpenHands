"""Pull request operations for Azure DevOps integration."""

from datetime import datetime

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.azure_devops.service.base import AzureDevOpsMixinBase
from openhands.integrations.service_types import Comment, RequestMethod


class AzureDevOpsPRsMixin(AzureDevOpsMixinBase):
    """Mixin for Azure DevOps pull request operations."""

    def _truncate_comment(self, comment: str, max_length: int = 1000) -> str:
        """Truncate comment to max length."""
        if len(comment) <= max_length:
            return comment
        return comment[:max_length] + '...'

    async def add_pr_thread(
        self,
        repository: str,
        pr_number: int,
        comment_text: str,
        status: str = 'active',
    ) -> dict:
        """Create a new thread (comment) in an Azure DevOps pull request.

        Azure DevOps uses 'threads' concept where each thread contains comments.
        This creates a new thread with a single comment for general PR discussion.

        API Reference: https://learn.microsoft.com/en-us/rest/api/azure/devops/git/pull-request-threads/create

        Args:
            repository: Repository name in format "organization/project/repo"
            pr_number: The pull request number
            comment_text: The comment text to post
            status: Thread status ('active', 'fixed', 'wontFix', 'closed', 'byDesign', 'pending')

        Returns:
            API response with created thread information

        Raises:
            HTTPException: If the API request fails
        """
        org, project, repo = self._parse_repository(repository)

        # URL-encode components to handle spaces and special characters
        org_enc = self._encode_url_component(org)
        project_enc = self._encode_url_component(project)
        repo_enc = self._encode_url_component(repo)

        url = f'{self.base_url}/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/pullrequests/{pr_number}/threads?api-version=7.1'

        # Create thread payload with a comment
        # Reference: https://learn.microsoft.com/en-us/rest/api/azure/devops/git/pull-request-threads/create
        payload = {
            'comments': [
                {
                    'parentCommentId': 0,
                    'content': comment_text,
                    'commentType': 1,  # 1 = text comment
                }
            ],
            'status': status,
        }

        response, _ = await self._make_request(
            url=url, params=payload, method=RequestMethod.POST
        )

        logger.info(f'Created PR thread in {repository}#{pr_number}')
        return response

    async def add_pr_comment_to_thread(
        self,
        repository: str,
        pr_number: int,
        thread_id: int,
        comment_text: str,
    ) -> dict:
        """Add a comment to an existing thread in an Azure DevOps pull request.

        API Reference: https://learn.microsoft.com/en-us/rest/api/azure/devops/git/pull-request-thread-comments/create

        Args:
            repository: Repository name in format "organization/project/repo"
            pr_number: The pull request number
            thread_id: The thread ID to add the comment to
            comment_text: The comment text to post

        Returns:
            API response with created comment information

        Raises:
            HTTPException: If the API request fails
        """
        org, project, repo = self._parse_repository(repository)

        # URL-encode components to handle spaces and special characters
        org_enc = self._encode_url_component(org)
        project_enc = self._encode_url_component(project)
        repo_enc = self._encode_url_component(repo)

        url = f'{self.base_url}/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/pullrequests/{pr_number}/threads/{thread_id}/comments?api-version=7.1'

        payload = {
            'content': comment_text,
            'parentCommentId': 1,  # Reply to the thread's root comment
            'commentType': 1,  # 1 = text comment
        }

        response, _ = await self._make_request(
            url=url, params=payload, method=RequestMethod.POST
        )

        logger.info(
            f'Added comment to thread {thread_id} in PR {repository}#{pr_number}'
        )
        return response

    async def get_pr_threads(self, repository: str, pr_number: int) -> list[dict]:
        """Get all threads (comment conversations) for a pull request.

        API Reference: https://learn.microsoft.com/en-us/rest/api/azure/devops/git/pull-request-threads/list

        Args:
            repository: Repository name in format "organization/project/repo"
            pr_number: The pull request number

        Returns:
            List of thread objects containing comments

        Raises:
            HTTPException: If the API request fails
        """
        org, project, repo = self._parse_repository(repository)

        # URL-encode components to handle spaces and special characters
        org_enc = self._encode_url_component(org)
        project_enc = self._encode_url_component(project)
        repo_enc = self._encode_url_component(repo)

        url = f'{self.base_url}/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/pullrequests/{pr_number}/threads?api-version=7.1'

        response, _ = await self._make_request(url)

        return response.get('value', [])

    async def get_pr_comments(
        self, repository: str, pr_number: int, max_comments: int = 100
    ) -> list[Comment]:
        """Get all comments from all threads in a pull request.

        Retrieves all threads and extracts comments from them, converting to Comment objects.

        Args:
            repository: Repository name in format "organization/project/repo"
            pr_number: The pull request number
            max_comments: Maximum number of comments to return

        Returns:
            List of Comment objects sorted by creation date
        """
        threads = await self.get_pr_threads(repository, pr_number)

        all_comments: list[Comment] = []

        for thread in threads:
            comments_data = thread.get('comments', [])

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

    async def create_pr(
        self,
        repo_name: str,
        source_branch: str,
        target_branch: str,
        title: str,
        body: str | None = None,
        draft: bool = False,
    ) -> str:
        """Creates a pull request in Azure DevOps.

        Args:
            repo_name: The repository name in format "organization/project/repo"
            source_branch: The source branch name
            target_branch: The target branch name
            title: The title of the pull request
            body: The description of the pull request
            draft: Whether to create a draft pull request

        Returns:
            The URL of the created pull request
        """
        # Parse repository string: organization/project/repo
        parts = repo_name.split('/')
        if len(parts) < 3:
            raise ValueError(
                f'Invalid repository format: {repo_name}. Expected format: organization/project/repo'
            )

        org = parts[0]
        project = parts[1]
        repo = parts[2]

        # URL-encode components to handle spaces and special characters
        org_enc = self._encode_url_component(org)
        project_enc = self._encode_url_component(project)
        repo_enc = self._encode_url_component(repo)

        url = f'https://dev.azure.com/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/pullrequests?api-version=7.1'

        # Set default body if none provided
        if not body:
            body = f'Merging changes from {source_branch} into {target_branch}'

        payload = {
            'sourceRefName': f'refs/heads/{source_branch}',
            'targetRefName': f'refs/heads/{target_branch}',
            'title': title,
            'description': body,
            'isDraft': draft,
        }

        response, _ = await self._make_request(
            url=url, params=payload, method=RequestMethod.POST
        )

        # Return the web URL of the created PR
        pr_id = response.get('pullRequestId')
        return f'https://dev.azure.com/{org_enc}/{project_enc}/_git/{repo_enc}/pullrequest/{pr_id}'

    async def get_pr_details(self, repository: str, pr_number: int) -> dict:
        """Get detailed information about a specific pull request.

        Args:
            repository: Repository name in Azure DevOps format 'org/project/repo'
            pr_number: The pull request number

        Returns:
            Raw API response from Azure DevOps
        """
        org, project, repo = self._parse_repository(repository)

        # URL-encode components to handle spaces and special characters
        org_enc = self._encode_url_component(org)
        project_enc = self._encode_url_component(project)
        repo_enc = self._encode_url_component(repo)

        url = f'{self.base_url}/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/pullrequests/{pr_number}?api-version=7.1'

        response, _ = await self._make_request(url)
        return response

    async def is_pr_open(self, repository: str, pr_number: int) -> bool:
        """Check if a PR is still active (not closed/merged).

        Args:
            repository: Repository name in Azure DevOps format 'org/project/repo'
            pr_number: The PR number to check

        Returns:
            True if PR is active (open), False if closed/merged/abandoned
        """
        try:
            pr_details = await self.get_pr_details(repository, pr_number)
            status = pr_details.get('status', '').lower()
            # Azure DevOps PR statuses: active, abandoned, completed
            return status == 'active'
        except Exception as e:
            logger.warning(
                f'Failed to check PR status for {repository}#{pr_number}: {e}'
            )
            return False

    async def add_pr_reaction(
        self, repository: str, pr_number: int, reaction_type: str = ':thumbsup:'
    ) -> dict:
        org, project, repo = self._parse_repository(repository)
        comment_text = f'{reaction_type} OpenHands is processing this PR...'
        return await self.add_pr_thread(
            repository, pr_number, comment_text, status='closed'
        )
