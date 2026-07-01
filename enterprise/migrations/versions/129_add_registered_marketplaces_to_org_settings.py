"""Add registered_marketplaces and updated_at columns to org table.

This column stores marketplace registrations for organization-level
plugin resolution. Composable with instance defaults and user marketplaces.
Also adds updated_at for optimistic locking.

Revision ID: 129
Revises: 128
Create Date: 2026-06-18 16:34:00.000

"""

from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '129'
down_revision: Union[str, None] = '128'
branch_labels: Union[str, None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    op.add_column('org', sa.Column('registered_marketplaces', sa.JSON(), nullable=True))
    op.add_column(
        'org',
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text('NOW()'),
        ),
    )


def downgrade() -> None:
    op.drop_column('org', 'updated_at')
    op.drop_column('org', 'registered_marketplaces')
