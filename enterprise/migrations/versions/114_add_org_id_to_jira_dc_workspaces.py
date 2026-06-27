"""add org_id to jira_dc_workspaces

Revision ID: 114
Revises: 113
Create Date: 2026-05-22 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '114'
down_revision: Union[str, None] = '113'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('jira_dc_workspaces', sa.Column('org_id', sa.UUID(), nullable=True))
    if op.get_bind().dialect.name == 'postgresql':
        op.execute(
            sa.text(
                """
                UPDATE jira_dc_workspaces AS j
                SET org_id = u.current_org_id
                FROM "user" AS u
                WHERE j.org_id IS NULL
                  AND j.admin_user_id ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
                  AND u.id = j.admin_user_id::uuid
                  AND u.current_org_id IS NOT NULL
                """
            )
        )
    op.create_foreign_key(
        'fk_jira_dc_workspaces_org_id',
        'jira_dc_workspaces',
        'org',
        ['org_id'],
        ['id'],
        ondelete='SET NULL',
    )


def downgrade() -> None:
    op.drop_constraint(
        'fk_jira_dc_workspaces_org_id',
        'jira_dc_workspaces',
        type_='foreignkey',
    )
    op.drop_column('jira_dc_workspaces', 'org_id')
