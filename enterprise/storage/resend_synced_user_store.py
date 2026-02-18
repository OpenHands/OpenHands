"""Store class for managing Resend synced users."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Optional, Set

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import sessionmaker
from storage.resend_synced_user import ResendSyncedUser


@dataclass
class ResendSyncedUserStore:
    """Store for tracking users synced to Resend audiences."""

    session_maker: sessionmaker

    def is_user_synced(self, email: str, audience_id: str) -> bool:
        """Check if a user has been synced to a specific audience.

        Args:
            email: The email address to check.
            audience_id: The Resend audience ID.

        Returns:
            True if the user has been synced, False otherwise.
        """
        with self.session_maker() as session:
            stmt = select(ResendSyncedUser).where(
                ResendSyncedUser.email == email.lower(),
                ResendSyncedUser.audience_id == audience_id,
            )
            result = session.execute(stmt).first()
            return result is not None

    def get_synced_emails_for_audience(self, audience_id: str) -> Set[str]:
        """Get all synced email addresses for a specific audience.

        Args:
            audience_id: The Resend audience ID.

        Returns:
            A set of lowercase email addresses that have been synced.
        """
        with self.session_maker() as session:
            stmt = select(ResendSyncedUser.email).where(
                ResendSyncedUser.audience_id == audience_id,
            )
            result = session.execute(stmt).scalars().all()
            return set(result)

    def mark_user_synced(
        self,
        email: str,
        audience_id: str,
        keycloak_user_id: Optional[str] = None,
    ) -> ResendSyncedUser:
        """Mark a user as synced to a specific audience.

        Uses upsert to handle race conditions - if the user is already
        marked as synced, this is a no-op.

        Args:
            email: The email address of the user.
            audience_id: The Resend audience ID.
            keycloak_user_id: Optional Keycloak user ID.

        Returns:
            The ResendSyncedUser record.

        Raises:
            RuntimeError: If the record could not be created or retrieved.
        """
        with self.session_maker() as session:
            stmt = (
                insert(ResendSyncedUser)
                .values(
                    email=email.lower(),
                    audience_id=audience_id,
                    keycloak_user_id=keycloak_user_id,
                    synced_at=datetime.now(UTC),
                )
                .on_conflict_do_nothing(constraint='uq_resend_synced_email_audience')
                .returning(ResendSyncedUser)
            )
            result = session.execute(stmt)
            session.commit()

            row = result.first()
            if row:
                return row[0]

            # on_conflict_do_nothing triggered, fetch the existing record
            existing = session.execute(
                select(ResendSyncedUser).where(
                    ResendSyncedUser.email == email.lower(),
                    ResendSyncedUser.audience_id == audience_id,
                )
            ).first()
            if existing:
                return existing[0]

            raise RuntimeError(
                f'Failed to create or retrieve synced user record for {email}'
            )

    def remove_synced_user(self, email: str, audience_id: str) -> bool:
        """Remove a user's synced status for a specific audience.

        Args:
            email: The email address of the user.
            audience_id: The Resend audience ID.

        Returns:
            True if a record was deleted, False if no record existed.
        """
        with self.session_maker() as session:
            stmt = delete(ResendSyncedUser).where(
                ResendSyncedUser.email == email.lower(),
                ResendSyncedUser.audience_id == audience_id,
            )
            result = session.execute(stmt)
            session.commit()
            return result.rowcount > 0
