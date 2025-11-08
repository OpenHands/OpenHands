"""create azure_devops_webhook table

Revision ID: 081
Revises: 080
Create Date: 2025-10-25 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '081'
down_revision: Union[str, None] = '080'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create Azure DevOps webhook table with proper constraints.

    Based on Azure DevOps Service Hook API structure:
    - organization/project/repository hierarchy
    - Webhooks can be at project-level (repository_id IS NULL) or repository-level
    - Uses composite keys for uniqueness at different levels
    """
    op.create_table(
        'azure_devops_webhook',
        sa.Column(
            'id', sa.Integer(), nullable=False, primary_key=True, autoincrement=True
        ),
        sa.Column('organization', sa.String(), nullable=False),
        sa.Column('project_id', sa.String(), nullable=False),
        sa.Column('repository_id', sa.String(), nullable=True),
        sa.Column('subscription_id', sa.String(), nullable=True),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('webhook_exists', sa.Boolean(), nullable=False),
        sa.Column('webhook_url', sa.String(), nullable=True),
        sa.Column('webhook_secret', sa.String(), nullable=True),
        sa.Column('webhook_uuid', sa.String(), nullable=True),
        sa.Column(
            'scopes',
            postgresql.ARRAY(sa.Text()),
            nullable=True,
        ),
        sa.Column(
            'last_synced',
            sa.DateTime(),
            server_default=sa.text('CURRENT_TIMESTAMP'),
            nullable=True,
        ),
    )

    # Create indexes for faster lookups
    op.create_index(
        'ix_azure_devops_webhook_user_id',
        'azure_devops_webhook',
        ['user_id'],
    )
    op.create_index(
        'ix_azure_devops_webhook_organization',
        'azure_devops_webhook',
        ['organization'],
    )
    op.create_index(
        'ix_azure_devops_webhook_project_id',
        'azure_devops_webhook',
        ['project_id'],
    )
    op.create_index(
        'ix_azure_devops_webhook_repository_id',
        'azure_devops_webhook',
        ['repository_id'],
    )
    op.create_index(
        'ix_azure_devops_webhook_subscription_id',
        'azure_devops_webhook',
        ['subscription_id'],
    )
    op.create_index(
        'ix_azure_devops_webhook_webhook_exists',
        'azure_devops_webhook',
        ['webhook_exists'],
    )
    op.create_index(
        'ix_azure_devops_webhook_last_synced',
        'azure_devops_webhook',
        ['last_synced'],
    )
    op.create_index(
        'ix_azure_devops_webhook_webhook_uuid',
        'azure_devops_webhook',
        ['webhook_uuid'],
    )

    # Create composite unique constraint for project-level webhooks (repository_id IS NULL)
    # This ensures one webhook per organization/project combination at project level
    op.create_index(
        'uq_azure_devops_webhook_project_level',
        'azure_devops_webhook',
        ['organization', 'project_id'],
        unique=True,
        postgresql_where=sa.text('repository_id IS NULL'),
    )

    # Create composite unique constraint for repository-level webhooks (repository_id IS NOT NULL)
    # This ensures one webhook per organization/project/repository combination
    op.create_index(
        'uq_azure_devops_webhook_repository_level',
        'azure_devops_webhook',
        ['organization', 'project_id', 'repository_id'],
        unique=True,
        postgresql_where=sa.text('repository_id IS NOT NULL'),
    )

    # Create unique constraint for subscription_id when it exists
    # Each Azure DevOps Service Hook subscription has a unique subscription_id
    op.create_index(
        'uq_azure_devops_webhook_subscription_id',
        'azure_devops_webhook',
        ['subscription_id'],
        unique=True,
        postgresql_where=sa.text('subscription_id IS NOT NULL'),
    )


def downgrade() -> None:
    """Drop the Azure DevOps webhook table and all associated indexes."""
    # Drop unique constraints first
    op.drop_index(
        'uq_azure_devops_webhook_subscription_id',
        table_name='azure_devops_webhook',
    )
    op.drop_index(
        'uq_azure_devops_webhook_repository_level',
        table_name='azure_devops_webhook',
    )
    op.drop_index(
        'uq_azure_devops_webhook_project_level',
        table_name='azure_devops_webhook',
    )

    # Drop regular indexes
    op.drop_index(
        'ix_azure_devops_webhook_webhook_uuid',
        table_name='azure_devops_webhook',
    )
    op.drop_index(
        'ix_azure_devops_webhook_last_synced',
        table_name='azure_devops_webhook',
    )
    op.drop_index(
        'ix_azure_devops_webhook_webhook_exists',
        table_name='azure_devops_webhook',
    )
    op.drop_index(
        'ix_azure_devops_webhook_subscription_id',
        table_name='azure_devops_webhook',
    )
    op.drop_index(
        'ix_azure_devops_webhook_repository_id',
        table_name='azure_devops_webhook',
    )
    op.drop_index(
        'ix_azure_devops_webhook_project_id',
        table_name='azure_devops_webhook',
    )
    op.drop_index(
        'ix_azure_devops_webhook_organization',
        table_name='azure_devops_webhook',
    )
    op.drop_index(
        'ix_azure_devops_webhook_user_id',
        table_name='azure_devops_webhook',
    )

    # Drop the table
    op.drop_table('azure_devops_webhook')
