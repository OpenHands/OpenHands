"""Add registered_marketplaces column to org table.

This column stores marketplace registrations for organization-level
plugin resolution. Composable with instance defaults and user marketplaces.

Revision ID: 125
Revises: 124
Create Date: 2026-06-18 16:34:00.000

"""

from typing import Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '125'
down_revision: Union[str, None] = '124'
branch_labels: Union[str, None] = None
depends_on: Union[str, None] = None


def upgrade() -> None:
    op.add_column(
        'org',
        sa.Column('registered_marketplaces', sa.JSON(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('org', 'registered_marketplaces')
