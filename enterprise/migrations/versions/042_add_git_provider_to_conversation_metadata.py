"""add git_provider to conversation_metadata

Revision ID: 042
Revises: 041
Create Date: 2025-05-29

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '042'
down_revision = '041'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'conversation_metadata', sa.Column('git_provider', sa.String(), nullable=True)
    )


def downgrade():
    op.drop_column('conversation_metadata', 'git_provider')
