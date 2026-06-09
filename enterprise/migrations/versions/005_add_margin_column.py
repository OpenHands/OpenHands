"""add margin column

Revision ID: 005
Revises: 004
Create Date: 2025-02-10 08:36:49.475467

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '005'
down_revision: Union[str, None] = '004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('settings', sa.Column('margin', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('settings', 'margin')
