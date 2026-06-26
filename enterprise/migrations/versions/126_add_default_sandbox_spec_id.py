"""Add default_sandbox_spec_id column to user and user_settings tables.

Revision ID: 125
Revises: 124
Create Date: 2026-06-05
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '126'
down_revision: str | None = '125'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'user',
        sa.Column('default_sandbox_spec_id', sa.String(), nullable=True),
    )
    op.add_column(
        'user_settings',
        sa.Column('default_sandbox_spec_id', sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('user_settings', 'default_sandbox_spec_id')
    op.drop_column('user', 'default_sandbox_spec_id')
