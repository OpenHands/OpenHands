from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from storage.database import a_session_maker
from storage.stored_repository import StoredRepository

from openhands.core.config.openhands_config import OpenHandsConfig


@dataclass
class RepositoryStore:
    config: OpenHandsConfig

    async def store_projects(self, repositories: list[StoredRepository]) -> None:
        """
        Store repositories in database (async version)

        1. Make sure to store repositories if its ID doesn't exist
        2. If repository ID already exists, make sure to only update the repo is_public and repo_name fields

        This implementation uses batch operations for better performance with large numbers of repositories.
        """
        if not repositories:
            return

        async with a_session_maker() as session:
            # Extract all repo_ids to check
            repo_ids = [r.repo_id for r in repositories]

            # Get all existing repositories in a single query
            result = await session.execute(
                select(StoredRepository).filter(StoredRepository.repo_id.in_(repo_ids))
            )
            existing_repos = {r.repo_id: r for r in result.scalars().all()}

            # Process all repositories
            for repo in repositories:
                if repo.repo_id in existing_repos:
                    # Update only is_public and repo_name fields for existing repositories
                    existing_repo = existing_repos[repo.repo_id]
                    existing_repo.is_public = repo.is_public
                    existing_repo.repo_name = repo.repo_name
                else:
                    # Add new repository to the session
                    session.add(repo)

            # Commit all changes
            await session.commit()

    @classmethod
    def get_instance(cls, config: OpenHandsConfig) -> RepositoryStore:
        """Get an instance of the UserRepositoryStore."""
        return RepositoryStore(config)
