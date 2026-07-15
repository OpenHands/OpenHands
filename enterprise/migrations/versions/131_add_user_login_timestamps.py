"""Add user login timestamps.

Revision ID: 131
Revises: 130
Create Date: 2026-06-16 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '131'
down_revision: Union[str, None] = '130'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('user', sa.Column('first_login_at', sa.DateTime(), nullable=True))
    op.add_column('user', sa.Column('last_login_at', sa.DateTime(), nullable=True))


def downgrade() -> None:
    op.drop_column('user', 'last_login_at')
    op.drop_column('user', 'first_login_at')
