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
    op.add_column(
        'event_callback',
        sa.Column(
            'updated_at', sa.DateTime, nullable=False, server_default=sa.func.now()
        ),
    )

    # Use batch_alter_table for SQLite compatibility when dropping/recreating columns
    with op.batch_alter_table('event_callback_result', schema=None) as batch_op:
        batch_op.drop_index('ix_event_callback_result_event_id')
        batch_op.drop_column('event_id')
        batch_op.add_column(sa.Column('event_id', sa.String, nullable=True))
        batch_op.create_index(
            batch_op.f('ix_event_callback_result_event_id'),
            ['event_id'],
            unique=False,
        )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('event_callback', 'status')
    op.drop_column('event_callback', 'updated_at')

    # Use batch_alter_table for SQLite compatibility when dropping/recreating columns
    with op.batch_alter_table('event_callback_result', schema=None) as batch_op:
        batch_op.drop_index('ix_event_callback_result_event_id')
        batch_op.drop_column('event_id')
        batch_op.add_column(sa.Column('event_id', sa.UUID, nullable=True))
        batch_op.create_index(
            batch_op.f('ix_event_callback_result_event_id'),
            ['event_id'],
            unique=False,
        )
