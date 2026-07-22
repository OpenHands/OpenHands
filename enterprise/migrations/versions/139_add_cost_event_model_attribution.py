"""Add model/usage attribution columns to conversation cost events.

Revision ID: 139
Revises: 138
Create Date: 2026-07-22

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '139'
down_revision = '138'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        'conversation_cost_events', sa.Column('usage_id', sa.String(), nullable=True)
    )
    op.add_column(
        'conversation_cost_events', sa.Column('llm_model', sa.String(), nullable=True)
    )
    op.add_column(
        'conversation_cost_events',
        sa.Column('prompt_tokens', sa.Integer(), nullable=True),
    )
    op.add_column(
        'conversation_cost_events',
        sa.Column('completion_tokens', sa.Integer(), nullable=True),
    )


def downgrade():
    op.drop_column('conversation_cost_events', 'completion_tokens')
    op.drop_column('conversation_cost_events', 'prompt_tokens')
    op.drop_column('conversation_cost_events', 'llm_model')
    op.drop_column('conversation_cost_events', 'usage_id')
