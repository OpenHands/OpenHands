"""Add tags column to conversation_metadata table

Revision ID: 008
Revises: 007
Create Date: 2026-03-31 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '008'
down_revision: Union[str, None] = '007'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add tags column for storing conversation metadata tags.

    Tags store key-value pairs for automation context, skills used, etc.
    The column is nullable for backwards compatibility with existing rows.
    """
    op.add_column(
        'conversation_metadata',
        sa.Column('tags', sa.JSON, nullable=True),
    )


def downgrade() -> None:
    """Remove tags column from conversation_metadata."""
    op.drop_column('conversation_metadata', 'tags')
