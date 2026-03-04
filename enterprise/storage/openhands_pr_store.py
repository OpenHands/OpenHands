from __future__ import annotations

from datetime import datetime

from sqlalchemy import and_, desc, select
from storage.database import a_session_maker
from storage.openhands_pr import OpenhandsPR

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.service_types import ProviderType


class OpenhandsPRStore:
    async def insert_pr(self, pr: OpenhandsPR) -> None:
        """
        Insert a new PR or delete and recreate if repo_id and pr_number already exist.
        """
        async with a_session_maker() as session:
            # Check if PR already exists
            result = await session.execute(
                select(OpenhandsPR).filter(
                    OpenhandsPR.repo_id == pr.repo_id,
                    OpenhandsPR.pr_number == pr.pr_number,
                    OpenhandsPR.provider == pr.provider,
                )
            )
            existing_pr = result.scalars().first()

            if existing_pr:
                # Delete existing PR
                await session.delete(existing_pr)
                await session.flush()

            session.add(pr)
            await session.commit()

    async def increment_process_attempts(self, repo_id: str, pr_number: int) -> bool:
        """
        Increment the process attempts counter for a PR.

        Args:
            repo_id: Repository identifier
            pr_number: Pull request number

        Returns:
            True if PR was found and updated, False otherwise
        """
        async with a_session_maker() as session:
            result = await session.execute(
                select(OpenhandsPR).filter(
                    OpenhandsPR.repo_id == repo_id, OpenhandsPR.pr_number == pr_number
                )
            )
            pr = result.scalars().first()

            if pr:
                pr.process_attempts += 1
                await session.merge(pr)
                await session.commit()
                return True
            return False

    async def update_pr_openhands_stats(
        self,
        repo_id: str,
        pr_number: int,
        original_updated_at: datetime,
        openhands_helped_author: bool,
        num_openhands_commits: int,
        num_openhands_review_comments: int,
        num_openhands_general_comments: int,
    ) -> bool:
        """
        Update OpenHands statistics for a PR with row-level locking and timestamp validation.

        Args:
            repo_id: Repository identifier
            pr_number: Pull request number
            original_updated_at: Original updated_at timestamp to check for concurrent modifications
            openhands_helped_author: Whether OpenHands helped the author (1+ commits)
            num_openhands_commits: Number of commits by OpenHands
            num_openhands_review_comments: Number of review comments by OpenHands
            num_openhands_general_comments: Number of PR comments (not review comments) by OpenHands

        Returns:
            True if PR was found and updated, False if not found or timestamp changed
        """
        async with a_session_maker() as session:
            # Use row-level locking to prevent concurrent modifications
            result = await session.execute(
                select(OpenhandsPR)
                .filter(
                    OpenhandsPR.repo_id == repo_id, OpenhandsPR.pr_number == pr_number
                )
                .with_for_update()
            )
            pr: OpenhandsPR | None = result.scalars().first()

            if not pr:
                # Current PR snapshot is stale
                logger.warning('Did not find PR {pr_number} for repo {repo_id}')
                return False

            # Check if the updated_at timestamp has changed (indicating concurrent modification)
            if pr.updated_at != original_updated_at:
                # Abort transaction - the PR was modified by another process
                await session.rollback()
                return False

            # Update the OpenHands statistics
            pr.openhands_helped_author = openhands_helped_author
            pr.num_openhands_commits = num_openhands_commits
            pr.num_openhands_review_comments = num_openhands_review_comments
            pr.num_openhands_general_comments = num_openhands_general_comments
            pr.processed = True

            await session.merge(pr)
            await session.commit()
            return True

    async def get_unprocessed_prs(
        self, limit: int = 50, max_retries: int = 3
    ) -> list[OpenhandsPR]:
        """
        Get unprocessed PR entries from the OpenhandsPR table.

        Args:
            limit: Maximum number of PRs to retrieve (default: 50)

        Returns:
            List of OpenhandsPR objects that need processing
        """
        async with a_session_maker() as session:
            result = await session.execute(
                select(OpenhandsPR)
                .filter(
                    and_(
                        ~OpenhandsPR.processed,
                        OpenhandsPR.process_attempts < max_retries,
                        OpenhandsPR.provider == ProviderType.GITHUB.value,
                    )
                )
                .order_by(desc(OpenhandsPR.updated_at))
                .limit(limit)
            )
            unprocessed_prs = list(result.scalars().all())

            return unprocessed_prs

    @classmethod
    def get_instance(cls) -> OpenhandsPRStore:
        """Get an instance of the OpenhandsPRStore."""
        return OpenhandsPRStore()
