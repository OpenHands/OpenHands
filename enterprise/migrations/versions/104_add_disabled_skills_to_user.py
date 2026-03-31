"""Add disabled_skills column to user table.

Migration 102 added disabled_skills to the legacy user_settings table,
but the active SaaS flow (SaasSettingsStore) reads from/writes to the
user table. This migration adds the column where it is actually needed.

Revision ID: 104
Revises: 103
Create Date: 2026-03-31
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '104'
down_revision: Union[str, None] = '103'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('user', sa.Column('disabled_skills', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('user', 'disabled_skills')
