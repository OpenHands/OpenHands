"""Drop experiment assignments table

Revision ID: 099
Revises: 098
Create Date: 2025-03-04

This migration drops the experiment_assignments table as the ExperimentManager
functionality has been removed from the codebase.
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = '099'
down_revision = '098'
branch_labels = None
depends_on = None


def upgrade():
    """Drop the experiment_assignments table."""
    op.drop_index(
        'ix_experiment_assignments_conversation_id', table_name='experiment_assignments'
    )
    op.drop_table('experiment_assignments')


def downgrade():
    """This is a one-way migration - the experiment functionality has been removed."""
    pass
