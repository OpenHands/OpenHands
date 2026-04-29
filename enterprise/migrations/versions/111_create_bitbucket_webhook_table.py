"""create bitbucket webhook table

Revision ID: 111
Revises: 110
Create Date: 2026-04-28 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '111'
down_revision: Union[str, None] = '110'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'bitbucket_webhook',
        sa.Column(
            'id', sa.Integer(), nullable=False, primary_key=True, autoincrement=True
        ),
        sa.Column('workspace', sa.String(), nullable=False),
        sa.Column('repo_slug', sa.String(), nullable=True),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('webhook_uuid', sa.String(), nullable=False),
        sa.Column('webhook_secret', sa.String(), nullable=True),
        sa.Column(
            'last_synced',
            sa.DateTime(),
            server_default=sa.text('CURRENT_TIMESTAMP'),
            nullable=True,
        ),
    )

    op.create_index('ix_bitbucket_webhook_user_id', 'bitbucket_webhook', ['user_id'])
    op.create_index(
        'ix_bitbucket_webhook_workspace', 'bitbucket_webhook', ['workspace']
    )
    op.create_unique_constraint(
        'uq_bitbucket_webhook_uuid', 'bitbucket_webhook', ['webhook_uuid']
    )


def downgrade() -> None:
    op.drop_constraint('uq_bitbucket_webhook_uuid', 'bitbucket_webhook', type_='unique')
    op.drop_index('ix_bitbucket_webhook_workspace', table_name='bitbucket_webhook')
    op.drop_index('ix_bitbucket_webhook_user_id', table_name='bitbucket_webhook')
    op.drop_table('bitbucket_webhook')
