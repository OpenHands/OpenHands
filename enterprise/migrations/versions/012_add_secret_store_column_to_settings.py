"""add secret_store column to settings table
Revision ID: 012
Revises: 011
Create Date: 2025-05-01 10:00:00.000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '012'
down_revision: Union[str, None] = '011'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'settings', sa.Column('secrets_store', sa.JSON(), nullable=True, default=False)
    )


def downgrade() -> None:
    op.drop_column('settings', 'secrets_store')
