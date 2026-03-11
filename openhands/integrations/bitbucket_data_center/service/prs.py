from typing import Any

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.bitbucket_data_center.service.base import (
    BitbucketDCMixinBase,
)
from openhands.integrations.service_types import RequestMethod


class BitbucketDCPRsMixin(BitbucketDCMixinBase):
    """
    Mixin for BitBucket data center pull request operations
    """

    async def create_pr(
        self,
        repo_name: str,
        source_branch: str,
        target_branch: str,
        title: str,
        body: str | None = None,
        draft: bool = False,
    ) -> str:
        """Creates a pull request in Bitbucket data center.

        Args:
            repo_name: The repository name in the format "project/repo"
            source_branch: The source branch name
            target_branch: The target branch name
            title: The title of the pull request
            body: The description of the pull request
            draft: Whether to create a draft pull request

        Returns:
            The URL of the created pull request
        """
        owner, repo = self._extract_owner_and_repo(repo_name)
        repo_base = self._repo_api_base(owner, repo)

        payload: dict[str, Any]

        url = f'{repo_base}/pull-requests'
        payload = {
            'title': title,
            'description': body or '',
            'fromRef': {
                'id': f'refs/heads/{source_branch}',
                'repository': {'slug': repo, 'project': {'key': owner}},
            },
            'toRef': {
                'id': f'refs/heads/{target_branch}',
                'repository': {'slug': repo, 'project': {'key': owner}},
            },
        }

        data, _ = await self._make_request(
            url=url, params=payload, method=RequestMethod.POST
        )

        # Return the URL to the pull request
        links = data.get('links', {}) if isinstance(data, dict) else {}

        if isinstance(links, dict):
            html_link = links.get('html')
            if isinstance(html_link, dict):
                href = html_link.get('href')
                if href:
                    return href
            if isinstance(html_link, list) and html_link:
                href = html_link[0].get('href')
                if href:
                    return href
            self_link = links.get('self')
            if isinstance(self_link, dict):
                href = self_link.get('href')
                if href:
                    return href
            if isinstance(self_link, list) and self_link:
                href = self_link[0].get('href')
                if href:
                    return href

        return ''

    async def get_pr_details(self, repository: str, pr_number: int) -> dict:
        """Get detailed information about a specific pull request

        Args:
            repository: Repository name in format 'owner/repo'
            pr_number: The pull request number

        Returns:
            Raw Bitbucket data center API response for the pull request
        """
        owner, repo = self._extract_owner_and_repo(repository)
        repo_base = self._repo_api_base(owner, repo)
        url = f'{repo_base}/pull-requests/{pr_number}'

        pr_data, _ = await self._make_request(url)

        return pr_data

    async def is_pr_open(self, repository: str, pr_number: int) -> bool:
        """Check if a Bitbucket data center pull request is still active (not closed/merged).

        Args:
            repository: Repository name in format 'owner/repo'
            pr_number: The PR number to check

        Returns:
            True if PR is active (OPEN), False if closed/merged
        """
        try:
            pr_details = await self.get_pr_details(repository, pr_number)

            # Bitbucket data center API response structure
            if 'state' in pr_details:
                # Bitbucket data center state values: OPEN, MERGED, DECLINED, SUPERSEDED
                return pr_details['state'] == 'OPEN'

            # If we can't determine the state, assume it's active (safer default)
            logger.warning(
                f'Could not determine Bitbucket PR status for {repository}#{pr_number}. '
                f'Response keys: {list(pr_details.keys())}. Assuming PR is active.'
            )
            return True

        except Exception as e:
            logger.warning(
                f'Could not determine Bitbucket PR status for {repository}#{pr_number}: {e}. '
                f'Including conversation to be safe.'
            )
            # If we can't determine the PR status, include the conversation to be safe
            return True
