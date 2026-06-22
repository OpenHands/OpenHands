"""Add registered_marketplaces and updated_at columns to user_settings table.

This column stores user's marketplace registrations for plugin resolution.
Also adds updated_at for optimistic locking.

Revision ID: 126
Revises: 125
Create Date: 2026-06-18 16:35:00.000000

"""

from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '126'
down_revision: Union[str, None] = '125'
branch_labels: Union[str, None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    op.add_column(
        'user_settings',
        sa.Column('registered_marketplaces', sa.JSON(), nullable=True)
    )
    op.add_column(
        'user_settings',
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('NOW()'),
        ),
    )


def downgrade() -> None:
    op.drop_column('user_settings', 'updated_at')
    op.drop_column('user_settings', 'registered_marketplaces')
