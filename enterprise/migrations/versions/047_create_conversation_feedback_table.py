"""Create conversation feedback table

Revision ID: 046
Revises: 045
Create Date: 2025-06-10

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '047'
down_revision = '046'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'conversation_feedback',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('conversation_id', sa.String(), nullable=False),
        sa.Column('event_id', sa.Integer(), nullable=True),
        sa.Column('rating', sa.Integer(), nullable=False),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column(
            'created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_conversation_feedback_conversation_id'),
        'conversation_feedback',
        ['conversation_id'],
        unique=False,
    )


def downgrade():
    op.drop_index(
        op.f('ix_conversation_feedback_conversation_id'),
        table_name='conversation_feedback',
    )
    op.drop_table('conversation_feedback')
