"""add parent_conversation_id to conversation_metadata

Revision ID: 003
Revises: 002
Create Date: 2025-11-06 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'conversation_metadata',
        sa.Column('parent_conversation_id', sa.String(), nullable=True),
    )
    op.create_index(
        op.f('ix_conversation_metadata_parent_conversation_id'),
        'conversation_metadata',
        ['parent_conversation_id'],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        op.f('ix_conversation_metadata_parent_conversation_id'),
        table_name='conversation_metadata',
    )
    op.drop_column('conversation_metadata', 'parent_conversation_id')
