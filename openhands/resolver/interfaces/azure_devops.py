# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import base64
import re
from typing import Any

import httpx

from openhands.resolver.interfaces.issue import (
    Issue,
    IssueHandlerInterface,
    ReviewThread,
)


class AzureDevOpsIssueHandler(IssueHandlerInterface):
    def __init__(
        self,
        token: str,
        organization: str,
        project: str,
        repository: str,
    ):
        self.token = token
        self.organization = organization
        self.project = project
        self.repository = repository
        self.owner = f'{organization}/{project}'
        self.base_api_url = f'https://dev.azure.com/{organization}/{project}/_apis'
        self.repo_api_url = f'{self.base_api_url}/git/repositories/{repository}'
        self.work_items_api_url = f'{self.base_api_url}/wit'
        self.default_branch = 'main'

    def set_owner(self, owner: str) -> None:
        """Set the owner of the repository."""
        self.owner = owner
        parts = owner.split('/')
        if len(parts) >= 2:
            self.organization = parts[0]
            self.project = parts[1]
            self.base_api_url = (
                f'https://dev.azure.com/{self.organization}/{self.project}/_apis'
            )
            self.repo_api_url = (
                f'{self.base_api_url}/git/repositories/{self.repository}'
            )
            self.work_items_api_url = f'{self.base_api_url}/wit'

    def get_headers(self) -> dict[str, str]:
        """Get the headers for the Azure DevOps API."""
        auth_str = base64.b64encode(f':{self.token}'.encode()).decode()
        return {
            'Authorization': f'Basic {auth_str}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }

    def download_issues(self) -> list[Any]:
        """Download issues from Azure DevOps."""
        # Use WIQL to query for active work items
        wiql_url = f'{self.work_items_api_url}/wiql?api-version=7.1'
        wiql_query = {
            'query': "SELECT [System.Id], [System.Title], [System.State] FROM WorkItems WHERE [System.State] = 'Active' ORDER BY [System.CreatedDate] DESC"
        }

        response = httpx.post(wiql_url, headers=self.get_headers(), json=wiql_query)
        response.raise_for_status()

        work_item_references = response.json().get('workItems', [])

        # Get details for each work item
        work_items = []
        for work_item_ref in work_item_references:
            work_item_id = work_item_ref.get('id')
            work_item_url = f'{self.work_items_api_url}/workitems/{work_item_id}?api-version=7.1&$expand=all'

            item_response = httpx.get(work_item_url, headers=self.get_headers())
            item_response.raise_for_status()

            work_items.append(item_response.json())

        return work_items

    def get_issue_comments(
        self, issue_number: int, comment_id: int | None = None
    ) -> list[str] | None:
        """Get comments for an issue."""
        comments_url = f'{self.work_items_api_url}/workitems/{issue_number}/comments?api-version=7.1-preview.3'

        response = httpx.get(comments_url, headers=self.get_headers())
        response.raise_for_status()

        comments_data = response.json().get('comments', [])

        if comment_id is not None:
            # Return a specific comment
            for comment in comments_data:
                if comment.get('id') == comment_id:
                    return [comment.get('text', '')]
            return None

        # Return all comments
        return [comment.get('text', '') for comment in comments_data]

    def get_base_url(self) -> str:
        """Get the base URL for the Azure DevOps repository."""
        return f'https://dev.azure.com/{self.organization}/{self.project}'

    def get_branch_url(self, branch_name: str) -> str:
        """Get the URL for a branch."""
        return f'{self.get_base_url()}/_git/{self.repository}?version=GB{branch_name}'

    def get_download_url(self) -> str:
        """Get the download URL for the repository."""
        return f'{self.get_base_url()}/_git/{self.repository}'

    def get_clone_url(self) -> str:
        """Get the clone URL for the repository."""
        return f'https://dev.azure.com/{self.organization}/{self.project}/_git/{self.repository}'

    def get_pull_url(self, pr_number: int) -> str:
        """Get the URL for a pull request."""
        return f'{self.get_base_url()}/_git/{self.repository}/pullrequest/{pr_number}'

    def get_graphql_url(self) -> str:
        """Get the GraphQL URL for Azure DevOps."""
        return f'https://dev.azure.com/{self.organization}/_apis/graphql?api-version=7.1-preview.1'

    def get_compare_url(self, branch_name: str) -> str:
        """Get the URL to compare branches."""
        return f'{self.get_base_url()}/_git/{self.repository}/branches?baseVersion=GB{self.default_branch}&targetVersion=GB{branch_name}&_a=files'

    def get_branch_name(self, base_branch_name: str) -> str:
        """Generate a branch name for a new pull request."""
        return f'openhands/issue-{base_branch_name}'

    def get_default_branch_name(self) -> str:
        """Get the default branch name for the repository."""
        # Get repository details to find the default branch
        response = httpx.get(
            f'{self.repo_api_url}?api-version=7.1', headers=self.get_headers()
        )
        response.raise_for_status()

        repo_data = response.json()
        default_branch = repo_data.get('defaultBranch', 'refs/heads/main')

        # Remove 'refs/heads/' prefix
        return default_branch.replace('refs/heads/', '')

    def branch_exists(self, branch_name: str) -> bool:
        """Check if a branch exists."""
        # List all branches and check if the branch exists
        response = httpx.get(
            f'{self.repo_api_url}/refs?filter=heads/{branch_name}&api-version=7.1',
            headers=self.get_headers(),
        )
        response.raise_for_status()

        refs = response.json().get('value', [])
        return len(refs) > 0

    def reply_to_comment(self, pr_number: int, comment_id: str, reply: str) -> None:
        """Reply to a comment on a pull request."""
        # Get the thread ID from the comment ID
        threads_url = (
            f'{self.repo_api_url}/pullRequests/{pr_number}/threads?api-version=7.1'
        )

        response = httpx.get(threads_url, headers=self.get_headers())
        response.raise_for_status()

        threads = response.json().get('value', [])
        thread_id = None

        for thread in threads:
            for comment in thread.get('comments', []):
                if str(comment.get('id')) == comment_id:
                    thread_id = thread.get('id')
                    break
            if thread_id:
                break

        if not thread_id:
            raise ValueError(f'Comment ID {comment_id} not found in PR {pr_number}')

        # Add a comment to the thread
        comment_url = f'{self.repo_api_url}/pullRequests/{pr_number}/threads/{thread_id}/comments?api-version=7.1'

        comment_data = {
            'content': reply,
            'parentCommentId': int(comment_id),
        }

        response = httpx.post(
            comment_url, headers=self.get_headers(), json=comment_data
        )
        response.raise_for_status()

    def send_comment_msg(self, issue_number: int, msg: str) -> None:
        """Send a comment to an issue."""
        comment_url = f'{self.work_items_api_url}/workitems/{issue_number}/comments?api-version=7.1-preview.3'

        comment_data = {
            'text': msg,
        }

        response = httpx.post(
            comment_url, headers=self.get_headers(), json=comment_data
        )
        response.raise_for_status()

    def get_authorize_url(self) -> str:
        """Get the authorization URL for Azure DevOps."""
        return 'https://app.vsaex.visualstudio.com/app/register'

    def create_pull_request(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Create a pull request."""
        if data is None:
            data = {}

        source_branch = data.get('source_branch')
        target_branch = data.get('target_branch', self.default_branch)
        title = data.get('title', 'Pull request created by OpenHands')
        description = data.get('description', '')

        pr_data = {
            'sourceRefName': f'refs/heads/{source_branch}',
            'targetRefName': f'refs/heads/{target_branch}',
            'title': title,
            'description': description,
        }

        response = httpx.post(
            f'{self.repo_api_url}/pullrequests?api-version=7.1',
            headers=self.get_headers(),
            json=pr_data,
        )
        response.raise_for_status()

        pr_response = response.json()

        return {
            'id': pr_response.get('pullRequestId'),
            'number': pr_response.get('pullRequestId'),
            'url': pr_response.get('url'),
        }

    def request_reviewers(self, reviewer: str, pr_number: int) -> None:
        """Request reviewers for a pull request."""
        # Get the reviewer's ID
        reviewer_url = f'https://vssps.dev.azure.com/{self.organization}/_apis/graph/users?api-version=7.1-preview.1'

        response = httpx.get(reviewer_url, headers=self.get_headers())
        response.raise_for_status()

        users = response.json().get('value', [])
        reviewer_id = None

        for user in users:
            if (
                user.get('displayName') == reviewer
                or user.get('mailAddress') == reviewer
            ):
                reviewer_id = user.get('descriptor')
                break

        if not reviewer_id:
            raise ValueError(f'Reviewer {reviewer} not found')

        # Add reviewer to the pull request
        reviewers_url = f'{self.repo_api_url}/pullRequests/{pr_number}/reviewers/{reviewer_id}?api-version=7.1'

        reviewer_data = {
            'vote': 0,  # No vote yet
        }

        response = httpx.put(
            reviewers_url, headers=self.get_headers(), json=reviewer_data
        )
        response.raise_for_status()

    def get_context_from_external_issues_references(
        self,
        closing_issues: list[str],
        closing_issue_numbers: list[int],
        issue_body: str,
        review_comments: list[str] | None,
        review_threads: list[ReviewThread],
        thread_comments: list[str] | None,
    ) -> list[str]:
        """Get context from external issue references."""
        context = []

        # Add issue body
        if issue_body:
            context.append(f'Issue description:\n{issue_body}')

        # Add thread comments
        if thread_comments:
            context.append('Thread comments:\n' + '\n'.join(thread_comments))

        # Add review comments
        if review_comments:
            context.append('Review comments:\n' + '\n'.join(review_comments))

        # Add review threads
        if review_threads:
            for thread in review_threads:
                context.append(
                    f'Review thread for files {", ".join(thread.files)}:\n{thread.comment}'
                )

        return context

    def get_converted_issues(
        self, issue_numbers: list[int] | None = None, comment_id: int | None = None
    ) -> list[Issue]:
        """Download issues from Azure DevOps and convert them to the Issue model."""
        if issue_numbers is None:
            # Download all issues
            work_items = self.download_issues()
        else:
            # Download specific issues
            work_items = []
            for issue_number in issue_numbers:
                work_item_url = f'{self.work_items_api_url}/workitems/{issue_number}?api-version=7.1&$expand=all'

                response = httpx.get(work_item_url, headers=self.get_headers())
                response.raise_for_status()

                work_items.append(response.json())

        issues = []
        for work_item in work_items:
            # Get basic issue information
            issue_number = work_item.get('id')
            title = work_item.get('fields', {}).get('System.Title', '')
            description = work_item.get('fields', {}).get('System.Description', '')

            # Get comments
            thread_comments = self.get_issue_comments(issue_number, comment_id)

            # Check if this is a pull request work item
            is_pr = False
            pr_number = None
            head_branch = None
            base_branch = None

            # Look for PR links in the work item relations
            for relation in work_item.get('relations', []):
                if relation.get(
                    'rel'
                ) == 'ArtifactLink' and 'pullrequest' in relation.get('url', ''):
                    is_pr = True
                    # Extract PR number from URL
                    pr_url = relation.get('url', '')
                    pr_match = re.search(r'pullRequests/(\d+)', pr_url)
                    if pr_match:
                        pr_number = int(pr_match.group(1))
                    break

            # If this is a PR, get the branch information
            if is_pr and pr_number:
                pr_url = f'{self.repo_api_url}/pullRequests/{pr_number}?api-version=7.1'

                pr_response = httpx.get(pr_url, headers=self.get_headers())
                pr_response.raise_for_status()

                pr_data = pr_response.json()
                head_branch = pr_data.get('sourceRefName', '').replace(
                    'refs/heads/', ''
                )
                base_branch = pr_data.get('targetRefName', '').replace(
                    'refs/heads/', ''
                )

                # Get PR review comments
                review_comments = []
                review_threads = []

                threads_url = f'{self.repo_api_url}/pullRequests/{pr_number}/threads?api-version=7.1'

                threads_response = httpx.get(threads_url, headers=self.get_headers())
                threads_response.raise_for_status()

                threads = threads_response.json().get('value', [])

                for thread in threads:
                    thread_comments = [
                        comment.get('content', '')
                        for comment in thread.get('comments', [])
                    ]
                    review_comments.extend(thread_comments)

                    # Get files associated with this thread
                    thread_files = []
                    if thread.get('threadContext', {}).get('filePath'):
                        thread_files.append(
                            thread.get('threadContext', {}).get('filePath')
                        )

                    if thread_comments:
                        review_threads.append(
                            ReviewThread(
                                comment='\n'.join(thread_comments),
                                files=thread_files,
                            )
                        )

            # Create the Issue object
            issue = Issue(
                owner=self.owner,
                repo=self.repository,
                number=issue_number,
                title=title,
                body=description,
                thread_comments=thread_comments,
                closing_issues=None,
                review_comments=review_comments if is_pr else None,
                review_threads=review_threads if is_pr else None,
                thread_ids=None,
                head_branch=head_branch,
                base_branch=base_branch,
            )

            issues.append(issue)

        return issues
