"""Pull request operations for Azure DevOps integration."""

from datetime import datetime
from typing import Any, Optional

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
            logger.error(f'Failed to create PR thread in {repository}#{pr_number}: {e}')
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
        thread_payload: dict[str, Any] = {
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
                f'Created new thread in {repository}#{pr_number}'\n                + (f' on {file_path}:{line_number}' if file_path else '')\n            )\n            return {'thread': response, 'comment': response.get('comments', [{}])[0]}\n        except Exception as e:\n            logger.error(\n                f'Failed to create new thread in {repository}#{pr_number}: {e}'\n            )\n            raise\n\n    async def get_pr_threads(self, repository: str, pr_number: int) -> list[dict]:\n        \"\"\"Get all threads (comment conversations) for a pull request.\n\n        API Reference: https://learn.microsoft.com/en-us/rest/api/azure/devops/git/pull-request-threads/list\n\n        Args:\n            repository: Repository name in format \"organization/project/repo\"\n            pr_number: The pull request number\n\n        Returns:\n            List of thread objects containing comments\n\n        Raises:\n            HTTPException: If the API request fails\n        \"\"\"\n        org, project, repo = self._parse_repository(repository)\n\n        # URL-encode components to handle spaces and special characters\n        org_enc = self._encode_url_component(org)\n        project_enc = self._encode_url_component(project)\n        repo_enc = self._encode_url_component(repo)\n\n        url = f'{self.base_url}/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/pullrequests/{pr_number}/threads?api-version=7.1'\n\n        try:\n            response, _ = await self._make_request(url)\n            return response.get('value', [])\n        except Exception as e:\n            logger.error(f'Failed to get PR threads for {repository}#{pr_number}: {e}')\n            raise\n\n    async def get_pr_comments(\n        self, repository: str, pr_number: int, max_comments: int = 100\n    ) -> list[Comment]:\n        \"\"\"Get all comments from all threads in a pull request.\n\n        Retrieves all threads and extracts comments from them, converting to\n        standardized Comment objects.\n\n        Args:\n            repository: Repository name in format \"organization/project/repo\"\n            pr_number: The pull request number\n            max_comments: Maximum number of comments to return (default: 100)\n\n        Returns:\n            List of Comment objects sorted by creation date\n\n        Raises:\n            HTTPException: If the API request fails\n        \"\"\"\n        threads = await self.get_pr_threads(repository, pr_number)\n\n        all_comments: list[Comment] = []\n\n        for thread in threads:\n            comments_data = thread.get('comments', [])\n\n            for comment_data in comments_data:\n                try:\n                    # Extract author information\n                    author_info = comment_data.get('author', {})\n                    author = author_info.get('displayName', 'unknown')\n\n                    # Parse dates\n                    created_at = self._parse_iso_datetime(\n                        comment_data.get('publishedDate')\n                    )\n                    updated_at = self._parse_iso_datetime(\n                        comment_data.get('lastUpdatedDate')\n                    )\n\n                    if updated_at is None:\n                        updated_at = created_at\n\n                    # Check if it's a system comment\n                    is_system = comment_data.get('commentType', 1) != 1  # 1 = text comment\n\n                    comment = Comment(\n                        id=str(comment_data.get('id', 0)),\n                        body=self._truncate_comment(\n                            comment_data.get('content', '')\n                        ),\n                        author=author,\n                        created_at=created_at,\n                        updated_at=updated_at,\n                        system=is_system,\n                    )\n\n                    all_comments.append(comment)\n                except Exception as e:\n                    logger.warning(\n                        f'Failed to parse comment {comment_data.get(\"id\")}: {e}'\n                    )\n                    continue\n\n        # Sort by creation date and limit\n        all_comments.sort(key=lambda c: c.created_at)\n        return all_comments[:max_comments]\n\n    @staticmethod\n    def _parse_iso_datetime(date_string: Optional[str]) -> datetime:\n        \"\"\"Parse ISO 8601 datetime string with Z timezone indicator.\n\n        Args:\n            date_string: ISO 8601 datetime string (may end with Z)\n\n        Returns:\n            Parsed datetime object, or epoch time if parsing fails\n        \"\"\"\n        if not date_string:\n            return datetime.fromtimestamp(0)\n\n        try:\n            # Replace Z with +00:00 for proper ISO 8601 parsing\n            normalized = date_string.replace('Z', '+00:00')\n            return datetime.fromisoformat(normalized)\n        except (ValueError, TypeError):\n            logger.warning(f'Failed to parse datetime: {date_string}')\n            return datetime.fromtimestamp(0)\n\n    async def create_pr(\n        self,\n        repo_name: str,\n        source_branch: str,\n        target_branch: str,\n        title: str,\n        body: Optional[str] = None,\n        draft: bool = False,\n        reviewers: Optional[list[str]] = None,\n        work_item_ids: Optional[list[int]] = None,\n    ) -> dict:\n        \"\"\"Creates a pull request in Azure DevOps.\n\n        Args:\n            repo_name: The repository name in format \"organization/project/repo\"\n            source_branch: The source branch name\n            target_branch: The target branch name\n            title: The title of the pull request\n            body: The description of the pull request (optional)\n            draft: Whether to create a draft pull request (default: False)\n            reviewers: Optional list of reviewer email addresses\n            work_item_ids: Optional list of work item IDs to link\n\n        Returns:\n            Dictionary containing the created PR details including 'pullRequestId' and 'url'\n\n        Raises:\n            ValueError: If repository format is invalid or required fields are missing\n            HTTPException: If the API request fails\n        \"\"\"\n        # Parse repository string: organization/project/repo\n        parts = repo_name.split('/')\n        if len(parts) < 3:\n            raise ValueError(\n                f'Invalid repository format: {repo_name}. Expected format: organization/project/repo'\n            )\n\n        org = parts[0]\n        project = parts[1]\n        repo = parts[2]\n\n        # Validate required fields\n        if not title or not title.strip():\n            raise ValueError('title cannot be empty')\n        if not source_branch or not source_branch.strip():\n            raise ValueError('source_branch cannot be empty')\n        if not target_branch or not target_branch.strip():\n            raise ValueError('target_branch cannot be empty')\n\n        # URL-encode components to handle spaces and special characters\n        org_enc = self._encode_url_component(org)\n        project_enc = self._encode_url_component(project)\n        repo_enc = self._encode_url_component(repo)\n\n        url = f'https://dev.azure.com/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/pullrequests?api-version=7.1'\n\n        # Set default body if none provided\n        if not body:\n            body = f'Merging changes from {source_branch} into {target_branch}'\n\n        payload = {\n            'sourceRefName': f'refs/heads/{source_branch}',\n            'targetRefName': f'refs/heads/{target_branch}',\n            'title': title,\n            'description': body,\n            'isDraft': draft,\n        }\n\n        # Add optional fields if provided\n        if reviewers:\n            payload['reviewers'] = [{'uniqueName': reviewer} for reviewer in reviewers]\n\n        if work_item_ids:\n            payload['workItemRefs'] = [\n                {'id': str(wid)} for wid in work_item_ids\n            ]\n\n        try:\n            response, _ = await self._make_request(\n                url=url, params=payload, method=RequestMethod.POST\n            )\n\n            # Return enhanced response with URL\n            pr_id = response.get('pullRequestId')\n            return {\n                'pullRequestId': pr_id,\n                'url': f'https://dev.azure.com/{org_enc}/{project_enc}/_git/{repo_enc}/pullrequest/{pr_id}',\n                **response,\n            }\n        except Exception as e:\n            logger.error(f'Failed to create PR in {repo_name}: {e}')\n            raise\n\n    async def get_pr_details(self, repository: str, pr_number: int) -> dict:\n        \"\"\"Get detailed information about a specific pull request.\n\n        Args:\n            repository: Repository name in Azure DevOps format 'org/project/repo'\n            pr_number: The pull request number\n\n        Returns:\n            Raw API response from Azure DevOps containing PR details\n\n        Raises:\n            ResourceNotFoundError: If PR doesn't exist\n            HTTPException: If the API request fails\n        \"\"\"\n        org, project, repo = self._parse_repository(repository)\n\n        # URL-encode components to handle spaces and special characters\n        org_enc = self._encode_url_component(org)\n        project_enc = self._encode_url_component(project)\n        repo_enc = self._encode_url_component(repo)\n\n        url = f'{self.base_url}/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/pullrequests/{pr_number}?api-version=7.1'\n\n        try:\n            response, _ = await self._make_request(url)\n            return response\n        except Exception as e:\n            if '404' in str(e) or 'not found' in str(e).lower():\n                raise ResourceNotFoundError(\n                    f'Pull request {repository}#{pr_number} not found'\n                )\n            logger.error(f'Failed to get PR details for {repository}#{pr_number}: {e}')\n            raise\n\n    async def is_pr_open(self, repository: str, pr_number: int) -> bool:\n        \"\"\"Check if a PR is still active (not closed/merged).\n\n        Args:\n            repository: Repository name in Azure DevOps format 'org/project/repo'\n            pr_number: The PR number to check\n\n        Returns:\n            True if PR is active (open), False if closed/merged/abandoned\n\n        Raises:\n            No exception - returns False on error to be safe\n        \"\"\"\n        try:\n            pr_details = await self.get_pr_details(repository, pr_number)\n            status = pr_details.get('status', '').lower()\n            # Azure DevOps PR statuses: active, abandoned, completed\n            return status == PRStatus.ACTIVE.lower()\n        except Exception as e:\n            logger.warning(\n                f'Failed to check PR status for {repository}#{pr_number}: {e}'\n            )\n            return False\n\n    async def add_pr_reaction(\n        self, repository: str, pr_number: int, reaction_type: str = ':thumbsup:'\n    ) -> dict:\n        \"\"\"Add a reaction comment to a pull request.\n\n        This creates a closed thread with a reaction comment, useful for\n        indicating status or acknowledgment without opening a discussion.\n\n        Args:\n            repository: Repository name in format \"organization/project/repo\"\n            pr_number: The pull request number\n            reaction_type: Emoji or reaction text (default: ':thumbsup:')\n\n        Returns:\n            API response with created thread information\n        \"\"\"\n        comment_text = f'{reaction_type} OpenHands is processing this PR...'\n        return await self.add_pr_thread(\n            repository, pr_number, comment_text, status=CommentThreadStatus.CLOSED\n        )\n\n    async def list_pull_requests(\n        self,\n        repository: str,\n        status: Optional[str] = None,\n        top: int = 10,\n        skip: int = 0,\n    ) -> dict:\n        \"\"\"List pull requests in a repository with optional filtering.\n\n        Args:\n            repository: Repository name in format \"organization/project/repo\"\n            status: Optional filter by status ('active', 'abandoned', 'completed')\n            top: Maximum number of results to return (default: 10)\n            skip: Number of results to skip for pagination (default: 0)\n\n        Returns:\n            Dictionary containing:\n            - 'count': Number of PRs returned\n            - 'value': List of PR objects\n            - 'hasMoreResults': Boolean indicating if more results exist\n\n        Raises:\n            ValueError: If status is invalid\n            HTTPException: If the API request fails\n        \"\"\"\n        if status:\n            self._validate_pr_status(status)\n\n        org, project, repo = self._parse_repository(repository)\n\n        org_enc = self._encode_url_component(org)\n        project_enc = self._encode_url_component(project)\n        repo_enc = self._encode_url_component(repo)\n\n        # Build query parameters\n        query_params = f'$top={top}&$skip={skip}'\n        if status:\n            # Map status string to Azure DevOps numeric value\n            status_map = {'active': 1, 'abandoned': 2, 'completed': 3}\n            query_params += f'&searchCriteria.status={status_map.get(status, 1)}'\n\n        url = f'{self.base_url}/{org_enc}/{project_enc}/_apis/git/repositories/{repo_enc}/pullrequests?{query_params}&api-version=7.1'\n\n        try:\n            response, _ = await self._make_request(url)\n            return {\n                'count': len(response.get('value', [])),\n                'value': response.get('value', []),\n                'hasMoreResults': response.get('continuationToken') is not None,\n            }\n        except Exception as e:\n            logger.error(f'Failed to list PRs for {repository}: {e}')\n            raise
