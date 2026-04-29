"""Add agent_kind column to conversation_metadata table

Revision ID: 009
Revises: 008
Create Date: 2026-04-27 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '009'
down_revision: Union[str, None] = '008'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    existing = [
        row[1]
        for row in bind.execute(sa.text('PRAGMA table_info(conversation_metadata)'))
    ]
    if 'agent_kind' not in existing:
        op.add_column(
            'conversation_metadata', sa.Column('agent_kind', sa.String, nullable=True)
        )


def downgrade() -> None:
    op.drop_column('conversation_metadata', 'agent_kind')
