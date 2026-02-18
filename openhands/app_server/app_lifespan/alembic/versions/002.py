"""Sync DB with Models

Revision ID: 001
Revises:
Create Date: 2025-10-05 11:28:41.772294

"""

from enum import Enum
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


class EventCallbackStatus(Enum):
    ACTIVE = 'ACTIVE'
    DISABLED = 'DISABLED'
    COMPLETED = 'COMPLETED'
    ERROR = 'ERROR'


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        'event_callback',
        sa.Column(
            'status',
            sa.Enum(EventCallbackStatus),
            nullable=False,
            server_default='ACTIVE',
        ),
    )
    # SQLite does not support ALTER TABLE ADD COLUMN with a non-constant default
    # (like CURRENT_TIMESTAMP) when NOT NULL. Use nullable=True for compatibility.
    op.add_column(
        'event_callback',
        sa.Column('updated_at', sa.DateTime, nullable=True),
    )
    # SQLite does not support ALTER TABLE DROP COLUMN directly via Alembic.
    # Use batch_alter_table to recreate the table with the new column type.
    with op.batch_alter_table('event_callback_result') as batch_op:
        batch_op.drop_index('ix_event_callback_result_event_id')
        batch_op.drop_column('event_id')
        batch_op.add_column(sa.Column('event_id', sa.String, nullable=True))
        batch_op.create_index(
            op.f('ix_event_callback_result_event_id'),
            ['event_id'],
            unique=False,
        )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('event_callback', 'status')
    op.drop_column('event_callback', 'updated_at')
    with op.batch_alter_table('event_callback_result') as batch_op:
        batch_op.drop_index('ix_event_callback_result_event_id')
        batch_op.drop_column('event_id')
        batch_op.add_column(sa.Column('event_id', sa.UUID, nullable=True))
        batch_op.create_index(
            op.f('ix_event_callback_result_event_id'),
            ['event_id'],
            unique=False,
        )
