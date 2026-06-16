"""Pull request operations for Azure DevOps integration."""

from datetime import datetime
from typing import Optional

from openhands.app_server.integrations.azure_devops.service.base import (
    AzureDevOpsMixinBase,
)
from openhands.app_server.integrations.service_types import (
    Comment,
    RequestMethod,
    ResourceNotFoundError,
)
from openhands.app_server.utils.logger import openhands_logger as logger


class CommentThreadStatus:
    """Azure DevOps comment thread status constants."""

    ACTIVE = 'active'
    FIXED = 'fixed'
    WONT_FIX = 'wontFix'
    CLOSED = 'closed'
    BY_DESIGN = 'byDesign'
    PENDING = 'pending'

    VALID_STATUSES = [ACTIVE, FIXED, WONT_FIX, CLOSED, BY_DESIGN, PENDING]


class PRStatus:
    """Azure DevOps pull request status constants."""

    ACTIVE = 'active'
    ABANDONED = 'abandoned'
    COMPLETED = 'completed'

    VALID_STATUSES = [ACTIVE, ABANDONED, COMPLETED]


class AzureDevOpsPRsMixin(AzureDevOpsMixinBase):
    """Mixin for Azure DevOps pull request operations."""

    def _truncate_comment(self, comment: str, max_length: int = 1000) -> str:
        """Truncate comment to max length.

        Args:
            comment: The comment text to truncate
            max_length: Maximum length (default: 1000)

        Returns:
            Truncated comment with ellipsis if exceeds max_length
        """
        if len(comment) <= max_length:
            return comment
        return comment[:max_length] + '...'

    def _validate_thread_status(self, status: str) -> None:
        """Validate that the thread status is valid.

        Args:
            status: The thread status to validate

        Raises:
            ValueError: If status is not a valid Azure DevOps thread status
        """
        if status not in CommentThreadStatus.VALID_STATUSES:
            raise ValueError(
                f"Invalid thread status '{status}'. "
                f"Valid statuses are: {', '.join(CommentThreadStatus.VALID_STATUSES)}"
            )

    def _validate_pr_status(self, status: str) -> None:
        """Validate that the PR status is valid.

        Args:
            status: The PR status to validate

        Raises:
            ValueError: If status is not a valid Azure DevOps PR status
        """
        if status not in PRStatus.VALID_STATUSES:
            raise ValueError(
                f"Invalid PR status '{status}'. "
                f"Valid statuses are: {', '.join(PRStatus.VALID_STATUSES)}"
            )

    async def add_pr_thread(
        self,
        repository: str,
        pr_number: int,
        comment_text: str,
        status: str = CommentThreadStatus.ACTIVE,
    ) -> dict:
        """Create a new thread (comment) in an Azure DevOps pull request.

        Azure DevOps uses 'threads' concept where each thread contains comments.
        This creates a new thread with a single comment for general PR discussion.

        API Reference: https://learn.microsoft.com/en-us/rest/api/azure/devops/git/pull-request-threads/create

        Args:
            repository: Repository name in format "organization/project/repo"
            pr_number: The pull request number
            comment_text: The comment text to post
            status: Thread status (default: 'active'). Valid values: 'active', 'fixed',
                   'wontFix', 'closed', 'byDesign', 'pending'

        Returns:
            API response with created thread information

        Raises:
            ValueError: If status is invalid
            HTTPException: If the API request fails
        """
        # Validate inputs
        if not comment_text or not comment_text.strip():
            raise ValueError('comment_text cannot be empty')
        self._validate_thread_status(status)

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

        try:
            response, _ = await self._make_request(
                url=url, params=payload, method=RequestMethod.POST
            )

            logger.info(f'Created PR thread in {repository}#{pr_number}')
            return response
        except Exception as e:
            logger.error(
                f'Failed to create PR thread in {repository}#{pr_number}: {e}'
            )
            raise

    async def add_pr_comment_to_thread(
        self,
        repository: str,
        pr_number: int,
        thread_id: int,
        comment_text: str,
    ) -> dict:
        """Add a comment to an existing thread in an Azure DevOps pull request.

        This method adds a reply to an existing comment thread, creating a
        threaded discussion within the PR.

        API Reference: https://learn.microsoft.com/en-us/rest/api/azure/devops/git/pull-request-thread-comments/create

        Args:
            repository: Repository name in format "organization/project/repo"
            pr_number: The pull request number
            thread_id: The thread ID to add the comment to
            comment_text: The comment text to post

        Returns:
            API response with created comment information

        Raises:
            ValueError: If comment_text is empty
            HTTPException: If the API request fails (e.g., thread not found)
        """
        # Validate inputs
        if not comment_text or not comment_text.strip():
            raise ValueError('comment_text cannot be empty')

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

        try:
            response, _ = await self._make_request(
                url=url, params=payload, method=RequestMethod.POST
            )

            logger.info(
                f'Added comment to thread {thread_id} in PR {repository}#{pr_number}'
            )
            return response
        except Exception as e:
            logger.error(
                f'Failed to add comment to thread {thread_id} in PR {repository}#{pr_number}: {e}'
            )
            raise

    async def add_pull_request_comment(
        self,
        repository: str,
        pr_number: int,
        content: str,
        thread_id: Optional[int] = None,
        parent_comment_id: Optional[int] = None,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        status: str = CommentThreadStatus.ACTIVE,
    ) -> dict:
        """Add a comment to a pull request with flexible options.

        This is a comprehensive method that supports three use cases:
        1. Reply to an existing thread (requires thread_id)
        2. Create a new thread with a comment in general discussion
        3. Create a file-level comment on a specific line

        Args:
            repository: Repository name in format "organization/project/repo"
            pr_number: The pull request number
            content: The comment content (required)
            thread_id: Optional - ID of existing thread to reply to
            parent_comment_id: Optional - ID of parent comment when replying to specific comment
            file_path: Optional - Path of the file to comment on (for file comments)
            line_number: Optional - Line number to comment on (for file comments)
            status: Thread status for new threads (default: 'active'). Only used when creating new threads.

        Returns:
            Dictionary containing:
            - For existing thread reply: {'comment': {...comment details...}}
            - For new thread: {'thread': {...thread details...}, 'comment': {...comment details...}}

        Raises:
            ValueError: If inputs are invalid
            HTTPException: If the API request fails

        Example:
            # Reply to existing thread
            result = await service.add_pull_request_comment(
                repository='org/project/repo',
                pr_number=42,
                content='I agree with this suggestion',
                thread_id=123
            )

            # Create new general discussion thread
            result = await service.add_pull_request_comment(
                repository='org/project/repo',
                pr_number=42,
                content='This looks good overall'
            )

            # Create file comment on specific line
            result = await service.add_pull_request_comment(
                repository='org/project/repo',
                pr_number=42,
                content='This variable name should be more descriptive',
                file_path='/src/app.ts',
                line_number=42
            )
        """
        # Validate inputs
        if not content or not content.strip():
            raise ValueError('content cannot be empty')

        self._validate_thread_status(status)

        # Case 1: Reply to existing thread
        if thread_id is not None:
            return await self._add_comment_to_existing_thread(
                repository=repository,
                pr_number=pr_number,
                thread_id=thread_id,
                content=content,
                parent_comment_id=parent_comment_id,
            )

        # Case 2 & 3: Create new thread (general discussion or file comment)
        return await self._create_new_thread_with_comment(
            repository=repository,
            pr_number=pr_number,
            content=content,
            status=status,
            file_path=file_path,
            line_number=line_number,
        )

    async def _add_comment_to_existing_thread(
        self,
        repository: str,
        pr_number: int,
        thread_id: int,
        content: str,
        parent_comment_id: Optional[int] = None,
    ) -> dict:
        """Internal method to add a comment to an existing thread.

        Args:
            repository: Repository name
            pr_number: Pull request number
            thread_id: ID of thread to reply to
            content: Comment content
            parent_comment_id: Optional parent comment ID for nested replies

        Returns:
            API response with created comment information
        """
        org, project, repo = self._parse_repository(repository)

        org_enc = self._encode_url_component(org)
        project_enc = self._encode_url_component(project)
        repo_enc = self._encode_url_component(repo)

        url = f'{self.base_url}/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/pullrequests/{pr_number}/threads/{thread_id}/comments?api-version=7.1'

        payload = {
            'content': content,
            'parentCommentId': parent_comment_id or 1,  # Default to thread root
            'commentType': 1,  # 1 = text comment
        }

        try:
            response, _ = await self._make_request(
                url=url, params=payload, method=RequestMethod.POST
            )
            logger.info(
                f'Added comment to thread {thread_id} in {repository}#{pr_number}'
            )
            return {'comment': response}
        except Exception as e:
            logger.error(
                f'Failed to add comment to thread {thread_id} in {repository}#{pr_number}: {e}'
            )
            raise

    async def _create_new_thread_with_comment(
        self,
        repository: str,
        pr_number: int,
        content: str,
        status: str,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
    ) -> dict:
        """Internal method to create a new thread with a comment.

        Args:
            repository: Repository name
            pr_number: Pull request number
            content: Comment content
            status: Thread status
            file_path: Optional file path for file comments
            line_number: Optional line number for file comments

        Returns:
            API response with created thread and comment information
        """
        org, project, repo = self._parse_repository(repository)

        org_enc = self._encode_url_component(org)
        project_enc = self._encode_url_component(project)
        repo_enc = self._encode_url_component(repo)

        url = f'{self.base_url}/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/pullrequests/{pr_number}/threads?api-version=7.1'

        # Build thread payload
        thread_payload = {
            'comments': [
                {
                    'parentCommentId': 0,
                    'content': content,
                    'commentType': 1,  # 1 = text comment
                }
            ],
            'status': status,
        }

        # Add file/line context if provided
        if file_path and line_number is not None:
            thread_payload['threadContext'] = {
                'filePath': file_path,
                'rightFileStart': {'line': line_number, 'offset': 1},
                'rightFileEnd': {'line': line_number, 'offset': 1},
            }

        try:
            response, _ = await self._make_request(
                url=url, params=thread_payload, method=RequestMethod.POST
            )
            logger.info(
                f'Created new thread in {repository}#{pr_number}'
                + (f' on {file_path}:{line_number}' if file_path else '')
            )
            return {'thread': response, 'comment': response.get('comments', [{}])[0]}
        except Exception as e:
            logger.error(
                f'Failed to create new thread in {repository}#{pr_number}: {e}'
            )
            raise

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

        try:
            response, _ = await self._make_request(url)
            return response.get('value', [])
        except Exception as e:
            logger.error(f'Failed to get PR threads for {repository}#{pr_number}: {e}')
            raise

    async def get_pr_comments(
        self, repository: str, pr_number: int, max_comments: int = 100
    ) -> list[Comment]:
        """Get all comments from all threads in a pull request.

        Retrieves all threads and extracts comments from them, converting to
        standardized Comment objects.

        Args:
            repository: Repository name in format "organization/project/repo"
            pr_number: The pull request number
            max_comments: Maximum number of comments to return (default: 100)

        Returns:
            List of Comment objects sorted by creation date

        Raises:
            HTTPException: If the API request fails
        """
        threads = await self.get_pr_threads(repository, pr_number)

        all_comments: list[Comment] = []

        for thread in threads:
            comments_data = thread.get('comments', [])

            for comment_data in comments_data:
                try:
                    # Extract author information
                    author_info = comment_data.get('author', {})
                    author = author_info.get('displayName', 'unknown')

                    # Parse dates
                    created_at = self._parse_iso_datetime(
                        comment_data.get('publishedDate')
                    )
                    updated_at = self._parse_iso_datetime(
                        comment_data.get('lastUpdatedDate')
                    )

                    if updated_at is None:
                        updated_at = created_at

                    # Check if it's a system comment
                    is_system = comment_data.get('commentType', 1) != 1  # 1 = text comment

                    comment = Comment(
                        id=str(comment_data.get('id', 0)),
                        body=self._truncate_comment(
                            comment_data.get('content', '')
                        ),
                        author=author,
                        created_at=created_at,
                        updated_at=updated_at,
                        system=is_system,
                    )

                    all_comments.append(comment)
                except Exception as e:
                    logger.warning(
                        f'Failed to parse comment {comment_data.get("id")}: {e}'
                    )
                    continue

        # Sort by creation date and limit
        all_comments.sort(key=lambda c: c.created_at)
        return all_comments[:max_comments]

    @staticmethod
    def _parse_iso_datetime(date_string: Optional[str]) -> datetime:
        """Parse ISO 8601 datetime string with Z timezone indicator.

        Args:
            date_string: ISO 8601 datetime string (may end with Z)

        Returns:
            Parsed datetime object, or epoch time if parsing fails
        """
        if not date_string:
            return datetime.fromtimestamp(0)

        try:
            # Replace Z with +00:00 for proper ISO 8601 parsing
            normalized = date_string.replace('Z', '+00:00')
            return datetime.fromisoformat(normalized)
        except (ValueError, TypeError):
            logger.warning(f'Failed to parse datetime: {date_string}')
            return datetime.fromtimestamp(0)

    async def create_pr(
        self,
        repo_name: str,
        source_branch: str,
        target_branch: str,
        title: str,
        body: Optional[str] = None,
        draft: bool = False,
        reviewers: Optional[list[str]] = None,
        work_item_ids: Optional[list[int]] = None,
    ) -> dict:
        """Creates a pull request in Azure DevOps.

        Args:
            repo_name: The repository name in format "organization/project/repo"
            source_branch: The source branch name
            target_branch: The target branch name
            title: The title of the pull request
            body: The description of the pull request (optional)
            draft: Whether to create a draft pull request (default: False)
            reviewers: Optional list of reviewer email addresses
            work_item_ids: Optional list of work item IDs to link

        Returns:
            Dictionary containing the created PR details including 'pullRequestId' and 'url'

        Raises:
            ValueError: If repository format is invalid or required fields are missing
            HTTPException: If the API request fails
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

        # Validate required fields
        if not title or not title.strip():
            raise ValueError('title cannot be empty')
        if not source_branch or not source_branch.strip():
            raise ValueError('source_branch cannot be empty')
        if not target_branch or not target_branch.strip():
            raise ValueError('target_branch cannot be empty')

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

        # Add optional fields if provided
        if reviewers:
            payload['reviewers'] = [{'uniqueName': reviewer} for reviewer in reviewers]

        if work_item_ids:
            payload['workItemRefs'] = [
                {'id': str(wid)} for wid in work_item_ids
            ]

        try:
            response, _ = await self._make_request(
                url=url, params=payload, method=RequestMethod.POST
            )

            # Return enhanced response with URL
            pr_id = response.get('pullRequestId')
            return {
                'pullRequestId': pr_id,
                'url': f'https://dev.azure.com/{org_enc}/{project_enc}/_git/{repo_enc}/pullrequest/{pr_id}',
                **response,
            }
        except Exception as e:
            logger.error(f'Failed to create PR in {repo_name}: {e}')
            raise

    async def get_pr_details(self, repository: str, pr_number: int) -> dict:
        """Get detailed information about a specific pull request.

        Args:
            repository: Repository name in Azure DevOps format 'org/project/repo'
            pr_number: The pull request number

        Returns:
            Raw API response from Azure DevOps containing PR details

        Raises:
            ResourceNotFoundError: If PR doesn't exist
            HTTPException: If the API request fails
        """
        org, project, repo = self._parse_repository(repository)

        # URL-encode components to handle spaces and special characters
        org_enc = self._encode_url_component(org)
        project_enc = self._encode_url_component(project)
        repo_enc = self._encode_url_component(repo)

        url = f'{self.base_url}/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/pullrequests/{pr_number}?api-version=7.1'

        try:
            response, _ = await self._make_request(url)
            return response
        except Exception as e:
            if '404' in str(e) or 'not found' in str(e).lower():
                raise ResourceNotFoundError(
                    f'Pull request {repository}#{pr_number} not found'
                )
            logger.error(
                f'Failed to get PR details for {repository}#{pr_number}: {e}'
            )
            raise

    async def is_pr_open(self, repository: str, pr_number: int) -> bool:
        """Check if a PR is still active (not closed/merged).

        Args:
            repository: Repository name in Azure DevOps format 'org/project/repo'
            pr_number: The PR number to check

        Returns:
            True if PR is active (open), False if closed/merged/abandoned

        Raises:
            No exception - returns False on error to be safe
        """
        try:
            pr_details = await self.get_pr_details(repository, pr_number)
            status = pr_details.get('status', '').lower()
            # Azure DevOps PR statuses: active, abandoned, completed
            return status == PRStatus.ACTIVE.lower()
        except Exception as e:
            logger.warning(
                f'Failed to check PR status for {repository}#{pr_number}: {e}'
            )
            return False

    async def add_pr_reaction(
        self, repository: str, pr_number: int, reaction_type: str = ':thumbsup:'
    ) -> dict:
        """Add a reaction comment to a pull request.

        This creates a closed thread with a reaction comment, useful for
        indicating status or acknowledgment without opening a discussion.

        Args:
            repository: Repository name in format "organization/project/repo"
            pr_number: The pull request number
            reaction_type: Emoji or reaction text (default: ':thumbsup:')

        Returns:
            API response with created thread information
        """
        comment_text = f'{reaction_type} OpenHands is processing this PR...'
        return await self.add_pr_thread(
            repository, pr_number, comment_text, status=CommentThreadStatus.CLOSED
        )

    async def list_pull_requests(
        self,
        repository: str,
        status: Optional[str] = None,
        top: int = 10,
        skip: int = 0,
    ) -> dict:
        """List pull requests in a repository with optional filtering.

        Args:
            repository: Repository name in format "organization/project/repo"
            status: Optional filter by status ('active', 'abandoned', 'completed')
            top: Maximum number of results to return (default: 10)
            skip: Number of results to skip for pagination (default: 0)

        Returns:
            Dictionary containing:
            - 'count': Number of PRs returned
            - 'value': List of PR objects
            - 'hasMoreResults': Boolean indicating if more results exist

        Raises:
            ValueError: If status is invalid
            HTTPException: If the API request fails
        """
        if status:
            self._validate_pr_status(status)

        org, project, repo = self._parse_repository(repository)

        org_enc = self._encode_url_component(org)
        project_enc = self._encode_url_component(project)
        repo_enc = self._encode_url_component(repo)

        # Build query parameters
        query_params = f'$top={top}&$skip={skip}'
        if status:
            # Map status string to Azure DevOps numeric value
            status_map = {'active': 1, 'abandoned': 2, 'completed': 3}
            query_params += f'&searchCriteria.status={status_map.get(status, 1)}'

        url = f'{self.base_url}/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/pullrequests?{query_params}&api-version=7.1'

        try:
            response, _ = await self._make_request(url)
            return {
                'count': len(response.get('value', [])),
                'value': response.get('value', []),
                'hasMoreResults': response.get('continuationToken') is not None,
            }
        except Exception as e:
            logger.error(f'Failed to list PRs for {repository}: {e}')
            raise
