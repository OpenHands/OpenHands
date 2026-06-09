"""add status column to gitlab-webhook table

Revision ID: 032
Revises: 031
Create Date: 2025-04-21

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '032'
down_revision: Union[str, None] = '031'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.rename_table('gitlab-webhook', 'gitlab_webhook')

    op.add_column(
        'gitlab_webhook',
        sa.Column(
            'last_synced',
            sa.DateTime(),
            server_default=sa.text('now()'),
            onupdate=sa.text('now()'),
            nullable=True,
        ),
    )

    op.drop_column('gitlab_webhook', 'webhook_name')

    op.alter_column(
        'gitlab_webhook',
        'scopes',
        existing_type=sa.String,
        type_=sa.ARRAY(sa.Text()),
        existing_nullable=True,
        postgresql_using='ARRAY[]::text[]',
    )


def downgrade() -> None:
    op.add_column(
        'gitlab_webhook', sa.Column('webhook_name', sa.Boolean(), nullable=True)
    )

    # Drop the new column from the renamed table
    op.drop_column('gitlab_webhook', 'last_synced')

    # Rename the table back
    op.rename_table('gitlab_webhook', 'gitlab-webhook')
