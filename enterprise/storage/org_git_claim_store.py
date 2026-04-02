"""
Store class for managing Git organization claims.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import and_, select
from storage.database import a_session_maker
from storage.org_git_claim import OrgGitClaim

from openhands.core.logger import openhands_logger as logger


class OrgGitClaimStore:
    """Store for managing Git organization claims."""

    @staticmethod
    async def create_claim(
        org_id: UUID,
        provider: str,
        git_organization: str,
        claimed_by: UUID,
    ) -> OrgGitClaim:
        """Create a new Git organization claim.

        Args:
            org_id: OpenHands organization UUID
            provider: Git provider ('github', 'gitlab', 'bitbucket')
            git_organization: Name of the Git organization being claimed
            claimed_by: User UUID who is making the claim

        Returns:
            OrgGitClaim: The created claim record
        """
        async with a_session_maker() as session:
            claim = OrgGitClaim(
                org_id=org_id,
                provider=provider,
                git_organization=git_organization,
                claimed_by=claimed_by,
                claimed_at=datetime.now(timezone.utc),
            )
            session.add(claim)
            await session.commit()
            await session.refresh(claim)

            logger.info(
                'Created Git organization claim',
                extra={
                    'claim_id': str(claim.id),
                    'org_id': str(org_id),
                    'provider': provider,
                    'git_organization': git_organization,
                    'claimed_by': str(claimed_by),
                },
            )

            return claim

    @staticmethod
    async def get_claim_by_provider_and_git_org(
        provider: str,
        git_organization: str,
    ) -> Optional[OrgGitClaim]:
        """Check if a Git organization is already claimed.

        Args:
            provider: Git provider name
            git_organization: Name of the Git organization

        Returns:
            OrgGitClaim or None if not claimed
        """
        async with a_session_maker() as session:
            result = await session.execute(
                select(OrgGitClaim).filter(
                    and_(
                        OrgGitClaim.provider == provider,
                        OrgGitClaim.git_organization == git_organization,
                    )
                )
            )
            return result.scalars().first()

    @staticmethod
    async def get_claims_by_org_id(org_id: UUID) -> list[OrgGitClaim]:
        """Get all Git organization claims for an OpenHands organization.

        Args:
            org_id: OpenHands organization UUID

        Returns:
            List of OrgGitClaim records
        """
        async with a_session_maker() as session:
            result = await session.execute(
                select(OrgGitClaim).filter(OrgGitClaim.org_id == org_id)
            )
            return list(result.scalars().all())

    @staticmethod
    async def delete_claim(claim_id: UUID, org_id: UUID) -> bool:
        """Delete a Git organization claim.

        Args:
            claim_id: Claim UUID to delete
            org_id: OpenHands organization UUID (for ownership verification)

        Returns:
            True if deleted, False if not found
        """
        async with a_session_maker() as session:
            result = await session.execute(
                select(OrgGitClaim).filter(
                    and_(
                        OrgGitClaim.id == claim_id,
                        OrgGitClaim.org_id == org_id,
                    )
                )
            )
            claim = result.scalars().first()

            if not claim:
                return False

            await session.delete(claim)
            await session.commit()

            logger.info(
                'Deleted Git organization claim',
                extra={
                    'claim_id': str(claim_id),
                    'org_id': str(org_id),
                    'provider': claim.provider,
                    'git_organization': claim.git_organization,
                },
            )

            return True
