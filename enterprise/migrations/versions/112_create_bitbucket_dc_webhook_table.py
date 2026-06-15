"""create bitbucket dc webhook table

Revision ID: 112
Revises: 111
Create Date: 2026-04-29 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '112'
down_revision: Union[str, None] = '111'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'bitbucket_dc_webhook',
        sa.Column(
            'id', sa.Integer(), nullable=False, primary_key=True, autoincrement=True
        ),
        sa.Column('project_key', sa.String(), nullable=False),
        sa.Column('repo_slug', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('webhook_id', sa.String(), nullable=True),
        sa.Column('webhook_secret', sa.String(), nullable=True),
        sa.Column(
            'last_synced',
            sa.DateTime(),
            server_default=sa.text('CURRENT_TIMESTAMP'),
            nullable=True,
        ),
    )

    op.create_index(
        'ix_bitbucket_dc_webhook_user_id', 'bitbucket_dc_webhook', ['user_id']
    )
    op.create_unique_constraint(
        'uq_bitbucket_dc_webhook_project_repo',
        'bitbucket_dc_webhook',
        ['project_key', 'repo_slug'],
    )


def downgrade() -> None:
    op.drop_constraint(
        'uq_bitbucket_dc_webhook_project_repo',
        'bitbucket_dc_webhook',
        type_='unique',
    )
    op.drop_index('ix_bitbucket_dc_webhook_user_id', table_name='bitbucket_dc_webhook')
    op.drop_table('bitbucket_dc_webhook')
