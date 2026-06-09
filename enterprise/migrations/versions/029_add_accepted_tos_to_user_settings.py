"""add accepted_tos to user_settings

Revision ID: 029
Revises: 028
Create Date: 2025-04-23

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '029'
down_revision: Union[str, None] = '028'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'user_settings', sa.Column('accepted_tos', sa.DateTime(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('user_settings', 'accepted_tos')
