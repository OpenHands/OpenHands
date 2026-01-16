"""Update conversation_metadata table to match StoredConversationMetadata dataclass

Revision ID: 005
Revises: 004
Create Date: 2025-11-11 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '005'
down_revision: Union[str, Sequence[str], None] = '004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    with op.batch_alter_table('conversation_metadata') as batch_op:
        # Drop columns not in StoredConversationMetadata dataclass
        batch_op.drop_column('github_user_id')

        # Alter user_id to become nullable
        batch_op.alter_column(
            'user_id',
            existing_type=sa.String(),
            nullable=True,
        )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('conversation_metadata') as batch_op:
        # Add back removed column
        batch_op.add_column(sa.Column('github_user_id', sa.String(), nullable=True))

        # Restore NOT NULL constraint
        batch_op.alter_column(
            'user_id',
            existing_type=sa.String(),
            nullable=False,
        )
