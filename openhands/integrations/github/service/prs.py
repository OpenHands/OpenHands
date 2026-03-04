from openhands.core.logger import openhands_logger as logger
from openhands.integrations.github.service.base import GitHubMixinBase
from openhands.integrations.service_types import RequestMethod


class GitHubPRsMixin(GitHubMixinBase):
    """Methods for interacting with GitHub PRs
    """

    async def create_pr(
        self,
        repo_name: str,
        source_branch: str,
        target_branch: str,
        title: str,
        body: str | None = None,
        draft: bool = True,
        labels: list[str] | None = None,
    ) -> str:
        """Creates a PR using user credentials

        Args:
            repo_name: The full name of the repository (owner/repo)
            source_branch: The name of the branch where your changes are implemented
            target_branch: The name of the branch you want the changes pulled into
            title: The title of the pull request (optional, defaults to a generic title)
            body: The body/description of the pull request (optional)
            draft: Whether to create the PR as a draft (optional, defaults to False)
            labels: A list of labels to apply to the pull request (optional)

        Returns:
            - PR URL when successful
            - Error message when unsuccessful
        """
        url = f'{self.BASE_URL}/repos/{repo_name}/pulls'

        # Set default body if none provided
        if not body:
            body = f'Merging changes from {source_branch} into {target_branch}'

        # Prepare the request payload
        payload = {
            'title': title,
            'head': source_branch,
            'base': target_branch,
            'body': body,
            'draft': draft,
        }

        # Make the POST request to create the PR
        response, _ = await self._make_request(
            url=url, params=payload, method=RequestMethod.POST
        )

        # Add labels if provided (PRs are a type of issue in GitHub's API)
        if labels and len(labels) > 0:
            pr_number = response['number']
            labels_url = f'{self.BASE_URL}/repos/{repo_name}/issues/{pr_number}/labels'
            labels_payload = {'labels': labels}
            await self._make_request(
                url=labels_url, params=labels_payload, method=RequestMethod.POST
            )

        # Return the HTML URL of the created PR
        return response['html_url']

    async def get_pr_details(self, repository: str, pr_number: int) -> dict:
        """Get detailed information about a specific pull request

        Args:
            repository: Repository name in format 'owner/repo'
            pr_number: The pull request number

        Returns:
            Raw GitHub API response for the pull request
        """
        url = f'{self.BASE_URL}/repos/{repository}/pulls/{pr_number}'
        pr_data, _ = await self._make_request(url)

        return pr_data

    async def is_pr_open(self, repository: str, pr_number: int) -> bool:
        """Check if a GitHub PR is still active (not closed/merged).

        Args:
            repository: Repository name in format 'owner/repo'
            pr_number: The PR number to check

        Returns:
            True if PR is active (open), False if closed/merged
        """
        try:
            pr_details = await self.get_pr_details(repository, pr_number)

            # GitHub API response structure
            # https://docs.github.com/en/rest/pulls/pulls#get-a-pull-request
            if 'state' in pr_details:
                return pr_details['state'] == 'open'
            elif 'merged' in pr_details and 'closed_at' in pr_details:
                # Check if PR is merged or closed
                return not (pr_details['merged'] or pr_details['closed_at'])

            # If we can't determine the state, assume it's active (safer default)
            logger.warning(
                f'Could not determine GitHub PR status for {repository}#{pr_number}. '
                f'Response keys: {list(pr_details.keys())}. Assuming PR is active.'
            )
            return True

        except Exception as e:
            logger.warning(
                f'Could not determine GitHub PR status for {repository}#{pr_number}: {e}. '
                f'Including conversation to be safe.'
            )
            # If we can't determine the PR status, include the conversation to be safe
            return True
