"""Add tags column to conversation_metadata table.

Tags store key-value pairs for automation context (trigger type, automation_id),
skills used, and other metadata. This enables querying conversations by
automation source and associating SDK-provided context with conversations.

Revision ID: 106
Revises: 105
Create Date: 2026-03-31
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '106'
down_revision: Union[str, None] = '105'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'conversation_metadata',
        sa.Column('tags', sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('conversation_metadata', 'tags')
