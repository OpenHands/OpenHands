"""Add mcp_config to org_member for user-specific MCP settings.

Revision ID: 103
Revises: 102
Create Date: 2026-03-26

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '103'
down_revision: Union[str, None] = '102'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('org_member', sa.Column('mcp_config', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('org_member', 'mcp_config')
