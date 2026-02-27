from openhands.core.logger import openhands_logger as logger
from openhands.integrations.bitbucket_data_center.service.base import (
    BitbucketDCMixinBase,
)
from openhands.integrations.service_types import ResourceNotFoundError
from openhands.microagent.types import MicroagentContentResponse, MicroagentResponse


class BitbucketDCFeaturesMixin(BitbucketDCMixinBase):
    """
    Mixin for BitBucket data center feature operations (microagents, cursor rules, etc.)
    """

    async def get_microagent_content(
        self, repository: str, file_path: str
    ) -> MicroagentContentResponse:
        """Fetch individual file content from Bitbucket data center repository.

        Args:
            repository: Repository name in format 'project/repo_slug'
            file_path: Path to the file within the repository

        Returns:
            MicroagentContentResponse with parsed content and triggers

        Raises:
            RuntimeError: If file cannot be fetched or doesn't exist
        """
        # Step 1: Get repository details using existing method
        repo_details = await self.get_repository_details_from_repo_name(repository)

        if not repo_details.main_branch:
            logger.warning(
                f'No main branch found in repository info for {repository}. '
                f'Repository response: mainbranch field missing'
            )
            raise ResourceNotFoundError(
                f'Main branch not found for repository {repository}. '
                f'This repository may be empty or have no default branch configured.'
            )

        # Step 2: Get file content using the main branch
        owner, repo = self._extract_owner_and_repo(repository)
        repo_base = self._repo_api_base(owner, repo)

        file_url = f'{repo_base}/browse/{file_path}'
        params = {'at': f'refs/heads/{repo_details.main_branch}'}
        response, _ = await self._make_request(file_url, params=params)
        if isinstance(response, dict):
            lines = response.get('lines')
            if isinstance(lines, list):
                content = '\n'.join(
                    line.get('text', '') for line in lines if isinstance(line, dict)
                )
            else:
                content = response.get('content', '')
        else:
            content = str(response)

        # Parse the content to extract triggers from frontmatter
        return self._parse_microagent_content(content, file_path)

    async def _process_microagents_directory(
        self, repository: str, microagents_path: str
    ) -> list[MicroagentResponse]:
        microagents = []
        try:
            directory_url = await self._get_microagents_directory_url(
                repository, microagents_path
            )
            directory_params = self._get_microagents_directory_params(microagents_path)
            response, _ = await self._make_request(directory_url, directory_params)

            # Bitbucket DC browse endpoint nests items under response['children']['values']
            items = response.get('children', {}).get('values', [])

            for item in items:
                if self._is_valid_microagent_file(item):
                    try:
                        file_name = self._get_file_name_from_item(item)
                        file_path = self._get_file_path_from_item(
                            item, microagents_path
                        )
                        microagents.append(
                            self._create_microagent_response(file_name, file_path)
                        )
                    except Exception as e:
                        logger.warning(f'Error processing microagent {item}: {str(e)}')
        except ResourceNotFoundError:
            logger.info(
                f'No microagents directory found in {repository} at {microagents_path}'
            )
        except Exception as e:
            logger.warning(f'Error fetching microagents directory: {str(e)}')

        return microagents
