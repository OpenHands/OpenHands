from storage.repository_store import RepositoryStore
from storage.stored_repository import StoredRepository
from storage.user_repo_map import UserRepositoryMap
from storage.user_repo_map_store import UserRepositoryMapStore

from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.core.logger import openhands_logger as logger
from openhands.integrations.service_types import Repository


async def store_repositories_in_db(repos: list[Repository], user_id: str) -> None:
    """
    Store repositories in DB and create user-repository mappings

    Args:
        repos: List of Repository objects to store
        user_id: User ID associated with these repositories
    """

    # Convert Repository objects to StoredRepository objects
    # Convert Repository objects to UserRepositoryMap objects
    stored_repos = []
    user_repos = []
    for repo in repos:
        repo_id = f'{repo.git_provider.value}##{str(repo.id)}'
        stored_repo = StoredRepository(
            repo_name=repo.full_name,
            repo_id=repo_id,
            is_public=repo.is_public,
            # Optional fields set to None by default
            has_microagent=None,
            has_setup_script=None,
        )
        stored_repos.append(stored_repo)
        user_repo_map = UserRepositoryMap(user_id=user_id, repo_id=repo_id, admin=None)

        user_repos.append(user_repo_map)

    # Get config instance
    config = OpenHandsConfig()

    try:
        # Store repositories in the repos table
        repo_store = RepositoryStore.get_instance(config)
        repo_store.store_projects(stored_repos)

        # Store user-repository mappings in the user-repos table
        user_repo_store = UserRepositoryMapStore.get_instance(config)
        user_repo_store.store_user_repo_mappings(user_repos)

        logger.info(f'Saved repos for user {user_id}')
    except Exception:
        logger.warning('Failed to save repos', exc_info=True)
