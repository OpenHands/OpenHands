from openhands.core.logger import openhands_logger as logger
from openhands.integrations.gitlab.service.base import GitLabMixinBase
from openhands.integrations.service_types import RequestMethod


class GitLabPRsMixin(GitLabMixinBase):
    """Methods for interacting with GitLab merge requests (PRs)
    """

    async def create_mr(
        self,
        id: int | str,
        source_branch: str,
        target_branch: str,
        title: str,
        description: str | None = None,
        labels: list[str] | None = None,
    ) -> str:
        """Creates a merge request in GitLab

        Args:
            id: The ID or URL-encoded path of the project
            source_branch: The name of the branch where your changes are implemented
            target_branch: The name of the branch you want the changes merged into
            title: The title of the merge request (optional, defaults to a generic title)
            description: The description of the merge request (optional)
            labels: A list of labels to apply to the merge request (optional)

        Returns:
            - MR URL when successful
            - Error message when unsuccessful
        """
        # Convert string ID to URL-encoded path if needed
        project_id = str(id).replace('/', '%2F') if isinstance(id, str) else id
        url = f'{self.BASE_URL}/projects/{project_id}/merge_requests'

        # Set default description if none provided
        if not description:
            description = f'Merging changes from {source_branch} into {target_branch}'

        # Prepare the request payload
        payload = {
            'source_branch': source_branch,
            'target_branch': target_branch,
            'title': title,
            'description': description,
        }

        # Add labels if provided
        if labels and len(labels) > 0:
            payload['labels'] = ','.join(labels)

        # Make the POST request to create the MR
        response, _ = await self._make_request(
            url=url, params=payload, method=RequestMethod.POST
        )

        return response['web_url']

    async def get_pr_details(self, repository: str, pr_number: int) -> dict:
        """Get detailed information about a specific merge request

        Args:
            repository: Repository name in format 'owner/repo'
            pr_number: The merge request number (iid)

        Returns:
            Raw GitLab API response for the merge request
        """
        project_id = self._extract_project_id(repository)
        url = f'{self.BASE_URL}/projects/{project_id}/merge_requests/{pr_number}'
        mr_data, _ = await self._make_request(url)

        return mr_data

    async def is_pr_open(self, repository: str, pr_number: int) -> bool:
        """Check if a GitLab merge request is still active (not closed/merged).

        Args:
            repository: Repository name in format 'owner/repo'
            pr_number: The merge request number (iid)

        Returns:
            True if MR is active (opened), False if closed/merged
        """
        try:
            mr_details = await self.get_pr_details(repository, pr_number)

            # GitLab API response structure
            # https://docs.gitlab.com/ee/api/merge_requests.html#get-single-mr
            if 'state' in mr_details:
                return mr_details['state'] == 'opened'
            elif 'merged_at' in mr_details and 'closed_at' in mr_details:
                # Check if MR is merged or closed
                return not (mr_details['merged_at'] or mr_details['closed_at'])

            # If we can't determine the state, assume it's active (safer default)
            logger.warning(
                f'Could not determine GitLab MR status for {repository}#{pr_number}. '
                f'Response keys: {list(mr_details.keys())}. Assuming MR is active.'
            )
            return True

        except Exception as e:
            logger.warning(
                f'Could not determine GitLab MR status for {repository}#{pr_number}: {e}. '
                f'Including conversation to be safe.'
            )
            # If we can't determine the MR status, include the conversation to be safe
            return True
