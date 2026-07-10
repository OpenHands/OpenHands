"""Add conversation cost events table.

Revision ID: 135
Revises: 134
Create Date: 2026-07-08

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '135'
down_revision = '134'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'conversation_cost_events',
        sa.Column('id', sa.Integer(), sa.Identity(), primary_key=True),
        sa.Column(
            'conversation_id',
            sa.String(),
            sa.ForeignKey('conversation_metadata.conversation_id'),
            nullable=False,
        ),
        sa.Column('cost_delta', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('occurred_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        'ix_conversation_cost_events_conversation_id',
        'conversation_cost_events',
        ['conversation_id'],
    )
    op.create_index(
        'ix_conversation_cost_events_occurred_at',
        'conversation_cost_events',
        ['occurred_at'],
    )


def downgrade():
    op.drop_index(
        'ix_conversation_cost_events_occurred_at',
        table_name='conversation_cost_events',
    )
    op.drop_index(
        'ix_conversation_cost_events_conversation_id',
        table_name='conversation_cost_events',
    )
    op.drop_table('conversation_cost_events')
