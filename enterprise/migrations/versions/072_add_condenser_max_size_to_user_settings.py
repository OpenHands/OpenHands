"""add condenser_max_size to user_settings

Revision ID: 072
Revises: 071
Create Date: 2025-08-26

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '072'
down_revision: Union[str, None] = '071'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'user_settings', sa.Column('condenser_max_size', sa.Integer(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('user_settings', 'condenser_max_size')
