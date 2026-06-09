"""add llm_model to conversation_metadata

Revision ID: 044
Revises: 043
Create Date: 2025-05-30

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '044'
down_revision = '043'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'conversation_metadata', sa.Column('llm_model', sa.String(), nullable=True)
    )


def downgrade():
    op.drop_column('conversation_metadata', 'llm_model')
