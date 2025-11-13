"""Update conversation_metadata table to match StoredConversationMetadata dataclass

Revision ID: 003
Revises: 002
Create Date: 2025-11-11 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, Sequence[str], None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Drop columns that are not in the StoredConversationMetadata dataclass
    op.drop_column('conversation_metadata', 'github_user_id')
    op.alter_column(
        'conversation_metadata',
        'user_id',
        existing_type=sa.String(),
        nullable=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Add back the dropped columns
    op.add_column(
        'conversation_metadata', sa.Column('github_user_id', sa.String(), nullable=True)
    )
    op.alter_column(
        'conversation_metadata',
        'user_id',
        existing_type=sa.String(),
        nullable=False,
    )
