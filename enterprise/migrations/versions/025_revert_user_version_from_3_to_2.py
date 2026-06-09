"""Revert user_version from 3 to 2

Revision ID: 025
Revises: 024
Create Date: 2025-04-09

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '025'
down_revision: Union[str, None] = '024'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Update user_version from 3 to 2 for all users who have version 3
    op.execute(
        """
        UPDATE user_settings
        SET user_version = 2
        WHERE user_version = 3
        """
    )


def downgrade() -> None:
    # Revert back to version 3 for users who have version 2
    # Note: This is not a perfect downgrade as we can't know which users originally had version 3
    op.execute(
        """
        UPDATE user_settings
        SET user_version = 3
        WHERE user_version = 2
        """
    )
