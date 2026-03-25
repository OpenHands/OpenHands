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
    conn = op.get_bind()
    orgs_with_config = conn.execute(
        sa.text('SELECT id, mcp_config FROM org WHERE mcp_config IS NOT NULL')
    ).fetchall()

    for org_id, mcp_config in orgs_with_config:
        conn.execute(
            sa.text(
                'UPDATE org_member SET mcp_config = :config WHERE org_id = :org_id'
            ),
            {'config': mcp_config, 'org_id': str(org_id)},
        )


def downgrade() -> None:
    op.drop_column('org_member', 'mcp_config')
