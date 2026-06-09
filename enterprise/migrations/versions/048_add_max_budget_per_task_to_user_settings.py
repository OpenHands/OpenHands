"""add max_budget_per_task to user_settings

Revision ID: 048
Revises: 047
Create Date: 2025-06-20

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '048'
down_revision: Union[str, None] = '047'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'user_settings', sa.Column('max_budget_per_task', sa.Float(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('user_settings', 'max_budget_per_task')
