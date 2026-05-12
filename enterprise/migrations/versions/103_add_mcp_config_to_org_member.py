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

    # Migrate existing org-level MCP configs to all members in each org.
    # This preserves existing configurations while transitioning to user-specific settings.
    # Uses server-side SQL to avoid pulling sensitive config data into the Python process.
    op.execute(
        sa.text(
            """
            UPDATE org_member
            SET mcp_config = org.mcp_config
            FROM org
            WHERE org_member.org_id = org.id
              AND org.mcp_config IS NOT NULL
            """
        )
    )


def downgrade() -> None:
    op.drop_column('org_member', 'mcp_config')
