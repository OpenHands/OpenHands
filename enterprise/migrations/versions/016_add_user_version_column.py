"""Add user settings version which acts as a hint of external db state

Revision ID: 016
Revises: 015
Create Date: 2025-03-20 16:30:00.000

"""

from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '016'
down_revision: Union[str, None] = '015'
branch_labels: Union[str, sa.Sequence[str], None] = None
depends_on: Union[str, sa.Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'user_settings',
        sa.Column('user_version', sa.Integer(), nullable=False, server_default='0'),
    )


def downgrade() -> None:
    op.drop_column('user_settings', 'user_version')
