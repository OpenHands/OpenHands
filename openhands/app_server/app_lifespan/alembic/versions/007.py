"""Add pending_messages table for server-side message queuing

Revision ID: 007
Revises: 006
Create Date: 2025-03-15 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '007'
down_revision: Union[str, None] = '006'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create pending_messages table for storing messages before conversation is ready.

    Messages are stored temporarily until the conversation becomes ready, then
    delivered and deleted regardless of success or failure.
    """
    op.create_table(
        'pending_messages',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('conversation_id', sa.String(), nullable=False, index=True),
        sa.Column('role', sa.String(20), nullable=False, server_default='user'),
        sa.Column('content', sa.JSON, nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    """Remove pending_messages table."""
    op.drop_table('pending_messages')
