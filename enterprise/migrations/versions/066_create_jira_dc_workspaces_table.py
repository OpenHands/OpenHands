"""create jira_dc_workspaces table

Revision ID: 066
Revises: 065
Create Date: 2025-07-08 10:04:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '066'
down_revision: Union[str, None] = '065'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'jira_dc_workspaces',
        sa.Column(
            'id', sa.Integer(), nullable=False, primary_key=True, autoincrement=True
        ),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('admin_user_id', sa.String(), nullable=False),
        sa.Column('webhook_secret', sa.String(), nullable=False),
        sa.Column('svc_acc_email', sa.String(), nullable=False),
        sa.Column('svc_acc_api_key', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(),
            server_default=sa.text('CURRENT_TIMESTAMP'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(),
            server_default=sa.text('CURRENT_TIMESTAMP'),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_table('jira_dc_workspaces')
