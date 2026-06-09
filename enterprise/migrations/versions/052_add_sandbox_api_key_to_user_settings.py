"""add sandbox_api_key to user_settings

Revision ID: 052
Revises: 051
Create Date: 2025-06-24

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '052'
down_revision: Union[str, None] = '051'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'user_settings', sa.Column('sandbox_api_key', sa.String(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('user_settings', 'sandbox_api_key')
