"""Update conversation_cost_events foreign key to cascade deletes.

Without ON DELETE CASCADE, deleting a row from conversation_metadata fails
with a ForeignKeyViolationError when conversation_cost_events rows still
reference it. The cost-delta stream is an audit trail whose running total
is already maintained on conversation_metadata.accumulated_cost, so
cascade-deleting the audit rows when the conversation is removed is safe.

Revision ID: 136
Revises: 135
Create Date: 2026-07-09

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = '136'
down_revision = '135'
branch_labels = None
depends_on = None


def upgrade():
    # Drop the existing foreign key constraint
    op.drop_constraint(
        'conversation_cost_events_conversation_id_fkey',
        'conversation_cost_events',
        type_='foreignkey',
    )

    # Add the new foreign key constraint with cascade delete
    op.create_foreign_key(
        'conversation_cost_events_conversation_id_fkey',
        'conversation_cost_events',
        'conversation_metadata',
        ['conversation_id'],
        ['conversation_id'],
        ondelete='CASCADE',
    )


def downgrade():
    # Drop the cascade delete foreign key constraint
    op.drop_constraint(
        'conversation_cost_events_conversation_id_fkey',
        'conversation_cost_events',
        type_='foreignkey',
    )

    # Recreate the original foreign key constraint without cascade delete
    op.create_foreign_key(
        'conversation_cost_events_conversation_id_fkey',
        'conversation_cost_events',
        'conversation_metadata',
        ['conversation_id'],
        ['conversation_id'],
    )
