"""Add git_user_name and git_user_email to user_settings

Revision ID: 062
Revises: 061
Create Date: 2025-08-06

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '062'
down_revision = '061'
branch_labels = None
depends_on = None


def upgrade():
    # Add git_user_name and git_user_email columns to user_settings table
    op.add_column(
        'user_settings', sa.Column('git_user_name', sa.String(), nullable=True)
    )
    op.add_column(
        'user_settings', sa.Column('git_user_email', sa.String(), nullable=True)
    )


def downgrade():
    # Drop git_user_name and git_user_email columns from user_settings table
    op.drop_column('user_settings', 'git_user_email')
    op.drop_column('user_settings', 'git_user_name')
