"""Add registered_marketplaces column to user_settings table.

This column stores user's marketplace registrations for plugin resolution.

Revision ID: 130
Revises: 129
Create Date: 2026-06-18 16:35:00.000000

"""

from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '130'
down_revision: Union[str, None] = '129'
branch_labels: Union[str, None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    op.add_column(
        'user_settings', sa.Column('registered_marketplaces', sa.JSON(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('user_settings', 'registered_marketplaces')
