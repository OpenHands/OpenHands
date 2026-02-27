"""Add git_user_name and git_user_email columns to user table.

Revision ID: 090
Revises: 089
Create Date: 2025-01-22
"""

import sqlalchemy as sa
from alembic import op

revision = '090'
down_revision = '089'


def upgrade() -> None:
    op.add_column(
        'user',
        sa.Column('git_user_name', sa.String, nullable=True),
    )
    op.add_column(
        'user',
        sa.Column('git_user_email', sa.String, nullable=True),
    )


def downgrade() -> None:
    op.drop_column('user', 'git_user_email')
    op.drop_column('user', 'git_user_name')
