"""create saas settings table

Revision ID: 027
Revises: 026
Create Date: 2025-01-27 20:08:58.360566

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '027'
down_revision: Union[str, None] = '026'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # This was created to match the settings object - in future some of these strings should probabyl
    # be replaced with enum types.
    op.create_table(
        'gitlab-webhook',
        sa.Column(
            'id', sa.Integer(), nullable=False, primary_key=True, autoincrement=True
        ),
        sa.Column('group_id', sa.String(), nullable=True),
        sa.Column('project_id', sa.String(), nullable=True),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('webhook_exists', sa.Boolean(), nullable=False),
        sa.Column('webhook_name', sa.Boolean(), nullable=True),
        sa.Column('webhook_url', sa.String(), nullable=True),
        sa.Column('webhook_secret', sa.String(), nullable=True),
        sa.Column('scopes', sa.String, nullable=True),
    )

    # Create indexes for faster lookups
    op.create_index('ix_gitlab_webhook_user_id', 'gitlab-webhook', ['user_id'])
    op.create_index('ix_gitlab_webhook_group_id', 'gitlab-webhook', ['group_id'])
    op.create_index('ix_gitlab_webhook_project_id', 'gitlab-webhook', ['project_id'])

    # Add unique constraints on group_id and project_id to support UPSERT operations
    op.create_unique_constraint(
        'uq_gitlab_webhook_group_id', 'gitlab-webhook', ['group_id']
    )
    op.create_unique_constraint(
        'uq_gitlab_webhook_project_id', 'gitlab-webhook', ['project_id']
    )


def downgrade() -> None:
    # Drop the constraints and indexes first before dropping the table
    op.drop_constraint('uq_gitlab_webhook_group_id', 'gitlab-webhook', type_='unique')
    op.drop_constraint('uq_gitlab_webhook_project_id', 'gitlab-webhook', type_='unique')
    op.drop_index('ix_gitlab_webhook_user_id', table_name='gitlab-webhook')
    op.drop_index('ix_gitlab_webhook_group_id', table_name='gitlab-webhook')
    op.drop_index('ix_gitlab_webhook_project_id', table_name='gitlab-webhook')
    op.drop_table('gitlab-webhook')
