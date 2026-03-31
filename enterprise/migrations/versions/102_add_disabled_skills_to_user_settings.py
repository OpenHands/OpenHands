"""Add disabled_skills to user_settings.

Revision ID: 102
Revises: 101
Create Date: 2026-02-25

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '102'
down_revision: Union[str, None] = '101'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'user_settings', sa.Column('disabled_skills', sa.JSON(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('user_settings', 'disabled_skills')
