"""Create recaptcha_logs table

Revision ID: 014
Revises: 013
Create Date: 2026-06-16 00:00:00.000000
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '014'
down_revision: str | None = '013'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create recaptcha_logs table for storing reCAPTCHA verification logs."""
    op.create_table(
        'recaptcha_logs',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('user_id', sa.String(), nullable=True),
        sa.Column('score', sa.Float(), nullable=True),
        sa.Column('token', sa.String(), nullable=True),
        sa.Column('action', sa.String(), nullable=True),
        sa.Column('hostname', sa.String(), nullable=True),
        sa.Column('apk_package_name', sa.String(), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_table('recaptcha_logs')
