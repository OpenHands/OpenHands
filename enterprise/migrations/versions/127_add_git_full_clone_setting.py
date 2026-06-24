"""Add git_full_clone setting to user settings.

Revision ID: 127
Revises: 126
Create Date: 2026-06-23 00:00:00.000000
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '127'
down_revision: str | None = '126'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'user',
        sa.Column('git_full_clone', sa.Boolean(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('user', 'git_full_clone')
