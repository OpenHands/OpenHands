"""add search_api_key to user_settings

Revision ID: 040
Revises: 039
Create Date: 2025-05-23

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '040'
down_revision: Union[str, None] = '039'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'user_settings', sa.Column('search_api_key', sa.String(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('user_settings', 'search_api_key')
