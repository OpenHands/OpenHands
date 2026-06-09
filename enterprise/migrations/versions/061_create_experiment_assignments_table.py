"""Create experiment assignments table

Revision ID: 061
Revises: 060
Create Date: 2025-07-29

This migration creates a table to track experiment assignments for conversations.
Each row represents one experiment assignment with experiment_name and variant columns.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '061'
down_revision = '060'
branch_labels = None
depends_on = None


def upgrade():
    """Create the experiment_assignments table."""
    op.create_table(
        'experiment_assignments',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('conversation_id', sa.String(), nullable=True),
        sa.Column('experiment_name', sa.String(), nullable=False),
        sa.Column('variant', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'conversation_id',
            'experiment_name',
            name='uq_experiment_assignments_conversation_experiment',
        ),
    )

    # Create index on conversation_id for efficient lookups
    op.create_index(
        'ix_experiment_assignments_conversation_id',
        'experiment_assignments',
        ['conversation_id'],
    )


def downgrade():
    """Drop the experiment_assignments table."""
    op.drop_index(
        'ix_experiment_assignments_conversation_id', table_name='experiment_assignments'
    )
    op.drop_table('experiment_assignments')
