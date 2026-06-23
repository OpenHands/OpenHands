"""Add git_full_clone to conversation metadata.

Revision ID: 013
Revises: 012
Create Date: 2026-06-23 00:00:00.000000
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '013'
down_revision: str | None = '012'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'conversation_metadata',
        sa.Column('git_full_clone', sa.Boolean(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('conversation_metadata', 'git_full_clone')
