import base64
import json
import os
import re
from datetime import datetime
from enum import Enum
from typing import Any

from github import Auth, Github, GithubIntegration
from integrations.github.github_view import (
    GithubIssue,
)
from integrations.github.queries import PR_QUERY_BY_NODE_ID
from integrations.models import Message
from integrations.types import PRStatus, ResolverViewInterface
from integrations.utils import HOST
from pydantic import SecretStr
from server.auth.constants import GITHUB_APP_CLIENT_ID, GITHUB_APP_PRIVATE_KEY
from storage.openhands_pr import OpenhandsPR
from storage.openhands_pr_store import OpenhandsPRStore

from openhands.core.config import load_openhands_config
from openhands.core.logger import openhands_logger as logger
from openhands.integrations.github.github_service import GithubServiceImpl
from openhands.integrations.service_types import ProviderType
from openhands.storage import get_file_store
from openhands.storage.locations import get_conversation_dir

config = load_openhands_config()
file_store = get_file_store(config.file_store, config.file_store_path)


COLLECT_GITHUB_INTERACTIONS = (
    os.getenv('COLLECT_GITHUB_INTERACTIONS', 'false') == 'true'
)


class TriggerType(str, Enum):
    ISSUE_LABEL = 'issue-label'
    ISSUE_COMMENT = 'issue-coment'
    PR_COMMENT_MACRO = 'label'
    INLINE_PR_COMMENT_MACRO = 'inline-label'


class GitHubDataCollector:
    """
    Saves data on Cloud Resolver Interactions

    1. We always save
        - Resolver trigger (comment or label)
        - Metadata (who started the job, repo name, issue number)

    2. We save data for the type of interaction
        a. For labelled issues, we save
            - {conversation_dir}/{conversation_id}/github_data/issue__{repo_name}_{issue_number}.json
                - issue number
                - trigger
                - metadata
                - body
                - title
                - comments

            - {conversation_dir}/{conversation_id}/github_data/pr__{repo_name}_{pr_number}.json
                - pr_number
                - metadata
                - body
                - title
                - commits/authors

    3. For all PRs that were opened with the resolver, we save
        - github_data/prs/{repo_name}_{pr_number}/data.json
            - pr_number
            - title
            - body
            - commits/authors
            - code diffs
            - merge status (either merged/closed)
    """

    def __init__(self):
        self.file_store = file_store
        self.issues_path = 'github_data/issue-{}-{}/data.json'
        self.matching_pr_path = 'github_data/pr-{}-{}/data.json'
        # self.full_saved_pr_path = 'github_data/prs/{}-{}/data.json'
        self.full_saved_pr_path = 'prs/github/{}-{}/data.json'
        self.github_integration = GithubIntegration(
            auth=Auth.AppAuth(GITHUB_APP_CLIENT_ID, GITHUB_APP_PRIVATE_KEY)
        )
        self.conversation_id = None

    async def _get_repo_node_id(self, repo_id: str, gh_client) -> str:
        """
        Get the new GitHub GraphQL node ID for a repository using the GitHub client.

        Args:
            repo_id: Numeric repository ID as string (e.g., "123456789")
            gh_client: SaaSGitHubService client with authentication

        Returns:
            New format node ID for GraphQL queries (e.g., "R_kgDOLfkiww")
        """
        try:
            return await gh_client.get_repository_node_id(repo_id)
        except Exception:
            # Fallback to old format if REST API fails
            node_string = f'010:Repository{repo_id}'
            return base64.b64encode(node_string.encode()).decode()

    def _create_file_name(
        self, path: str, repo_id: str, number: int, conversation_id: str | None
    ):
        suffix = path.format(repo_id, number)

        if conversation_id:
            return f'{get_conversation_dir(conversation_id)}{suffix}'

        return suffix

    def _get_installation_access_token(self, installation_id: str) -> str:
        token_data = self.github_integration.get_access_token(
            installation_id  # type: ignore[arg-type]
        )
        return token_data.token

    def _check_openhands_author(self, name, login) -> bool:
        return (
            name == 'openhands'
            or login == 'openhands'
            or login == 'openhands-agent'
            or login == 'openhands-ai'
            or login == 'openhands-staging'
            or login == 'openhands-exp'
            or (login and 'openhands' in login.lower())
        )

    def _get_issue_comments(
        self, installation_id: str, repo_name: str, issue_number: int, conversation_id
    ) -> list[dict[str, Any]]:
        """
        Retrieve all comments from an issue until a comment with conversation_id is found
        """

        try:
            installation_token = self._get_installation_access_token(installation_id)

            with Github(auth=Auth.Token(installation_token)) as github_client:
                repo = github_client.get_repo(repo_name)
                issue = repo.get_issue(issue_number)
                comments = []

                for comment in issue.get_comments():
                    comment_data = {
                        'id': comment.id,
                        'body': comment.body,
                        'created_at': comment.created_at.isoformat(),
                        'user': comment.user.login,
                    }

                    # If we find a comment containing conversation_id, stop collecting comments
                    if conversation_id in comment.body:
                        break

                    comments.append(comment_data)

                return comments
        except Exception:
            return []

    def _save_data(self, path: str, data: dict[str, Any]):
        """Save data to a path"""
        self.file_store.write(path, json.dumps(data))

    def _save_issue(
        self,
        github_view: GithubIssue,
        trigger_type: TriggerType,
    ) -> None:
        """
        Save issue data when it's labeled with openhands

            1. Save under {conversation_dir}/{conversation_id}/github_data/issue_{issue_number}.json
            2. Save issue snapshot (title, body, comments)
            3. Save trigger type (label)
            4. Save PR opened (if exists, this information comes later when agent has finished its task)
                - Save commit shas
                - Save author info
            5. Was PR merged or closed
        """

        conversation_id = github_view.conversation_id

        if not conversation_id:
            return

        issue_number = github_view.issue_number
        file_name = self._create_file_name(
            path=self.issues_path,
            repo_id=github_view.full_repo_name,
            number=issue_number,
            conversation_id=conversation_id,
        )

        payload_data = github_view.raw_payload.message.get('payload', {})
        isssue_details = payload_data.get('issue', {})
        is_repo_private = payload_data.get('repository', {}).get('private', 'true')
        title = isssue_details.get('title', '')
        body = isssue_details.get('body', '')

        # Get comments for the issue
        comments = self._get_issue_comments(
            github_view.installation_id,
            github_view.full_repo_name,
            issue_number,
            conversation_id,
        )

        data = {
            'trigger': trigger_type,
            'metadata': {
                'user': github_view.user_info.username,
                'repo_name': github_view.full_repo_name,
                'is_repo_private': is_repo_private,
                'number': issue_number,
            },
            'contents': {
                'title': title,
                'body': body,
                'comments': comments,
            },
        }

        self._save_data(file_name, data)
        logger.info(
            f'[Github]: Saved issue #{issue_number} for {github_view.full_repo_name}'
        )

    def _get_pr_commits(self, installation_id: str, repo_name: str, pr_number: int):
        commits = []
        installation_token = self._get_installation_access_token(installation_id)
        with Github(auth=Auth.Token(installation_token)) as github_client:
            repo = github_client.get_repo(repo_name)
            pr = repo.get_pull(pr_number)

            for commit in pr.get_commits():
                commit_data = {
                    'sha': commit.sha,
                    'authors': commit.author.login if commit.author else None,
                    'committed_date': commit.commit.committer.date.isoformat()
                    if commit.commit and commit.commit.committer
                    else None,
                }
                commits.append(commit_data)

        return commits

    def _extract_repo_metadata(self, repo_data: dict) -> dict:
        """Extract repository metadata from GraphQL response"""
        return {
            'name': repo_data.get('name'),
            'owner': repo_data.get('owner', {}).get('login'),
            'languages': [
                lang['name'] for lang in repo_data.get('languages', {}).get('nodes', [])
            ],
        }

    def _process_commits_page(self, pr_data: dict, commits: list) -> None:
        """Process commits from a single GraphQL page"""
        commit_nodes = pr_data.get('commits', {}).get('nodes', [])
        for commit_node in commit_nodes:
            commit = commit_node['commit']
            author_info = commit.get('author', {})
            commit_data = {
                'sha': commit['oid'],
                'message': commit['message'],
                'committed_date': commit.get('committedDate'),
                'author': {
                    'name': author_info.get('name'),
                    'email': author_info.get('email'),
                    'github_login': author_info.get('user', {}).get('login'),
                },
                'stats': {
                    'additions': commit.get('additions', 0),
                    'deletions': commit.get('deletions', 0),
                    'changed_files': commit.get('changedFiles', 0),
                },
            }
            commits.append(commit_data)

    def _process_pr_comments_page(self, pr_data: dict, pr_comments: list) -> None:
        """Process PR comments from a single GraphQL page"""
        comment_nodes = pr_data.get('comments', {}).get('nodes', [])
        for comment in comment_nodes:
            comment_data = {
                'author': comment.get('author', {}).get('login'),
                'body': comment.get('body'),
                'created_at': comment.get('createdAt'),
                'type': 'pr_comment',
            }
            pr_comments.append(comment_data)

    def _process_review_comments_page(
        self, pr_data: dict, review_comments: list
    ) -> None:
        """Process reviews and review comments from a single GraphQL page"""
        review_nodes = pr_data.get('reviews', {}).get('nodes', [])
        for review in review_nodes:
            # Add the review itself if it has a body
            if review.get('body', '').strip():
                review_data = {
                    'author': review.get('author', {}).get('login'),
                    'body': review.get('body'),
                    'created_at': review.get('createdAt'),
                    'state': review.get('state'),
                    'type': 'review',
                }
                review_comments.append(review_data)

            # Add individual review comments
            review_comment_nodes = review.get('comments', {}).get('nodes', [])
            for review_comment in review_comment_nodes:
                review_comment_data = {
                    'author': review_comment.get('author', {}).get('login'),
                    'body': review_comment.get('body'),
                    'created_at': review_comment.get('createdAt'),
                    'type': 'review_comment',
                }
                review_comments.append(review_comment_data)

    def _count_openhands_activity(
        self, commits: list, review_comments: list, pr_comments: list
    ) -> tuple[int, int, int]:
        """Count OpenHands commits, review comments, and general PR comments"""
        openhands_commit_count = 0
        openhands_review_comment_count = 0
        openhands_general_comment_count = 0

        # Count commits by OpenHands (check both name and login)
        for commit in commits:
            author = commit.get('author', {})
            author_name = author.get('name', '').lower()
            author_login = (
                author.get('github_login', '').lower()
                if author.get('github_login')
                else ''
            )

            if self._check_openhands_author(author_name, author_login):
                openhands_commit_count += 1

        # Count review comments by OpenHands
        for review_comment in review_comments:
            author_login = (
                review_comment.get('author', '').lower()
                if review_comment.get('author')
                else ''
            )
            author_name = ''  # Initialize to avoid reference before assignment
            if self._check_openhands_author(author_name, author_login):
                openhands_review_comment_count += 1

        # Count general PR comments by OpenHands
        for pr_comment in pr_comments:
            author_login = (
                pr_comment.get('author', '').lower() if pr_comment.get('author') else ''
            )
            author_name = ''  # Initialize to avoid reference before assignment
            if self._check_openhands_author(author_name, author_login):
                openhands_general_comment_count += 1

        return (
            openhands_commit_count,
            openhands_review_comment_count,
            openhands_general_comment_count,
        )

    def _build_final_data_structure(
        self,
        repo_data: dict,
        pr_data: dict,
        commits: list,
        pr_comments: list,
        review_comments: list,
        openhands_commit_count: int,
        openhands_review_comment_count: int,
        openhands_general_comment_count: int = 0,
    ) -> dict:
        """Build the final data structure for JSON storage"""

        is_merged = pr_data['merged']
        merged_by = None
        merge_commit_sha = None
        if is_merged:
            merged_by = (pr_data.get('mergedBy') or {}).get('login')
            merge_commit_sha = (pr_data.get('mergeCommit') or {}).get('oid')

        return {
            'repo_metadata': self._extract_repo_metadata(repo_data),
            'pr_metadata': {
                'username': (pr_data.get('author') or {}).get('login'),
                'number': pr_data.get('number'),
                'title': pr_data.get('title'),
                'body': pr_data.get('body'),
                'comments': pr_comments,
            },
            'commits': commits,
            'review_comments': review_comments,
            'merge_status': {
                'merged': pr_data.get('merged'),
                'merged_by': merged_by,
                'state': pr_data.get('state'),
                'merge_commit_sha': merge_commit_sha,
            },
            'openhands_stats': {
                'num_commits': openhands_commit_count,
                'num_review_comments': openhands_review_comment_count,
                'num_general_comments': openhands_general_comment_count,
                'helped_author': openhands_commit_count > 0,
            },
        }

    async def save_full_pr(self, openhands_pr: OpenhandsPR) -> None:
        """
        Save PR information including metadata and commit details using GraphQL

        Saves:
        - Repo metadata (repo name, languages, contributors)
        - PR metadata (number, title, body, author, comments)
        - Commit information (sha, authors, message, stats)
        - Merge status
        - Num openhands commits
        - Num openhands review comments
        """
        pr_number = openhands_pr.pr_number
        installation_id = openhands_pr.installation_id
        repo_id = openhands_pr.repo_id

        # Get installation token and create Github client
        # This will fail if the user decides to revoke OpenHands' access to their repo
        # In this case, we will simply return when the exception occurs
        # This will not lead to infinite loops when processing PRs as we log number of attempts and cap max attempts independently from this
        try:
            installation_token = self._get_installation_access_token(installation_id)
        except Exception as e:
            logger.warning(
                f'Failed to generate token for {openhands_pr.repo_name}: {e}'
            )
            return

        gh_client = GithubServiceImpl(token=SecretStr(installation_token))

        # Get the new format GraphQL node ID
        node_id = await self._get_repo_node_id(repo_id, gh_client)

        # Initialize data structures
        commits: list[dict] = []
        pr_comments: list[dict] = []
        review_comments: list[dict] = []
        pr_data = None
        repo_data = None

        # Pagination cursors
        commits_after = None
        comments_after = None
        reviews_after = None

        # Fetch all data with pagination
        while True:
            variables = {
                'nodeId': node_id,
                'pr_number': pr_number,
                'commits_after': commits_after,
                'comments_after': comments_after,
                'reviews_after': reviews_after,
            }

            try:
                result = await gh_client.execute_graphql_query(
                    PR_QUERY_BY_NODE_ID, variables
                )
                if not result.get('data', {}).get('node', {}).get('pullRequest'):
                    break

                pr_data = result['data']['node']['pullRequest']
                repo_data = result['data']['node']

                # Process data from this page using modular methods
                self._process_commits_page(pr_data, commits)
                self._process_pr_comments_page(pr_data, pr_comments)
                self._process_review_comments_page(pr_data, review_comments)

                # Check pagination for all three types
                has_more_commits = (
                    pr_data.get('commits', {})
                    .get('pageInfo', {})
                    .get('hasNextPage', False)
                )
                has_more_comments = (
                    pr_data.get('comments', {})
                    .get('pageInfo', {})
                    .get('hasNextPage', False)
                )
                has_more_reviews = (
                    pr_data.get('reviews', {})
                    .get('pageInfo', {})
                    .get('hasNextPage', False)
                )

                # Update cursors
                if has_more_commits:
                    commits_after = (
                        pr_data.get('commits', {}).get('pageInfo', {}).get('endCursor')
                    )
                else:
                    commits_after = None

                if has_more_comments:
                    comments_after = (
                        pr_data.get('comments', {}).get('pageInfo', {}).get('endCursor')
                    )
                else:
                    comments_after = None

                if has_more_reviews:
                    reviews_after = (
                        pr_data.get('reviews', {}).get('pageInfo', {}).get('endCursor')
                    )
                else:
                    reviews_after = None

                # Continue if there's more data to fetch
                if not (has_more_commits or has_more_comments or has_more_reviews):
                    break

            except Exception:
                logger.warning('Error fetching PR data', exc_info=True)
                return

        if not pr_data or not repo_data:
            return

        # Count OpenHands activity using modular method
        (
            openhands_commit_count,
            openhands_review_comment_count,
            openhands_general_comment_count,
        ) = self._count_openhands_activity(commits, review_comments, pr_comments)

        logger.info(
            f'[Github]: PR #{pr_number} - OpenHands commits: {openhands_commit_count}, review comments: {openhands_review_comment_count}, general comments: {openhands_general_comment_count}'
        )
        logger.info(
            f'[Github]: PR #{pr_number} - Total collected: {len(commits)} commits, {len(pr_comments)} PR comments, {len(review_comments)} review comments'
        )

        # Build final data structure using modular method
        data = self._build_final_data_structure(
            repo_data,
            pr_data or {},
            commits,
            pr_comments,
            review_comments,
            openhands_commit_count,
            openhands_review_comment_count,
            openhands_general_comment_count,
        )

        # Update the OpenhandsPR object with OpenHands statistics
        store = OpenhandsPRStore.get_instance()
        openhands_helped_author = openhands_commit_count > 0

        # Update the PR with OpenHands statistics
        update_success = store.update_pr_openhands_stats(
            repo_id=repo_id,
            pr_number=pr_number,
            original_updated_at=openhands_pr.updated_at,
            openhands_helped_author=openhands_helped_author,
            num_openhands_commits=openhands_commit_count,
            num_openhands_review_comments=openhands_review_comment_count,
            num_openhands_general_comments=openhands_general_comment_count,
        )

        if not update_success:
            logger.warning(
                f'[Github]: Failed to update OpenHands stats for PR #{pr_number} in repo {repo_id} - PR may have been modified concurrently'
            )

        # Save to file
        file_name = self._create_file_name(
            path=self.full_saved_pr_path,
            repo_id=repo_id,
            number=pr_number,
            conversation_id=None,
        )
        self._save_data(file_name, data)
        logger.info(
            f'[Github]: Saved full PR #{pr_number} for repo {repo_id} with OpenHands stats: commits={openhands_commit_count}, reviews={openhands_review_comment_count}, general_comments={openhands_general_comment_count}, helped={openhands_helped_author}'
        )

    def _check_for_conversation_url(self, body):
        conversation_pattern = re.search(
            rf'https://{HOST}/conversations/([a-zA-Z0-9-]+)(?:\s|[.,;!?)]|$)', body
        )
        if conversation_pattern:
            return conversation_pattern.group(1)

        return None

    def _is_pr_closed_or_merged(self, payload):
        """
        Check if PR was closed (regardless of conversation URL)
        """
        action = payload.get('action', '')
        return action == 'closed' and 'pull_request' in payload

    def _track_closed_or_merged_pr(self, payload):
        """
        Track PR closed/merged event
        """

        repo_id = str(payload['repository']['id'])
        pr_number = payload['number']
        installation_id = str(payload['installation']['id'])
        private = payload['repository']['private']
        repo_name = payload['repository']['full_name']

        pr_data = payload['pull_request']

        # Extract PR metrics
        num_reviewers = len(pr_data.get('requested_reviewers', []))
        num_commits = pr_data.get('commits', 0)
        num_review_comments = pr_data.get('review_comments', 0)
        num_general_comments = pr_data.get('comments', 0)
        num_changed_files = pr_data.get('changed_files', 0)
        num_additions = pr_data.get('additions', 0)
        num_deletions = pr_data.get('deletions', 0)
        merged = pr_data.get('merged', False)

        # Extract closed_at timestamp
        # Example: "closed_at":"2025-06-19T21:19:36Z"
        closed_at_str = pr_data.get('closed_at')
        created_at = pr_data.get('created_at')

        closed_at = datetime.fromisoformat(closed_at_str.replace('Z', '+00:00'))

        # Determine status based on whether it was merged
        status = PRStatus.MERGED if merged else PRStatus.CLOSED

        store = OpenhandsPRStore.get_instance()

        pr = OpenhandsPR(
            repo_name=repo_name,
            repo_id=repo_id,
            pr_number=pr_number,
            status=status,
            provider=ProviderType.GITHUB.value,
            installation_id=installation_id,
            private=private,
            num_reviewers=num_reviewers,
            num_commits=num_commits,
            num_review_comments=num_review_comments,
            num_changed_files=num_changed_files,
            num_additions=num_additions,
            num_deletions=num_deletions,
            merged=merged,
            created_at=created_at,
            closed_at=closed_at,
            # These properties will be enriched later
            openhands_helped_author=None,
            num_openhands_commits=None,
            num_openhands_review_comments=None,
            num_general_comments=num_general_comments,
        )

        store.insert_pr(pr)
        logger.info(f'Tracked PR {status}: {repo_id}#{pr_number}')

    def process_payload(self, message: Message):
        if not COLLECT_GITHUB_INTERACTIONS:
            return

        raw_payload = message.message.get('payload', {})

        if self._is_pr_closed_or_merged(raw_payload):
            self._track_closed_or_merged_pr(raw_payload)

    async def save_data(self, github_view: ResolverViewInterface):
        if not COLLECT_GITHUB_INTERACTIONS:
            return

        return

        # TODO: track issue metadata in DB and save comments to filestore
