"""Update conversation_callbacks foreign key to cascade deletes

Revision ID: 051
Revises: 050
Create Date: 2025-06-24

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = '051'
down_revision = '050'
branch_labels = None
depends_on = None


def upgrade():
    # Drop the existing foreign key constraint
    op.drop_constraint(
        'conversation_callbacks_conversation_id_fkey',
        'conversation_callbacks',
        type_='foreignkey',
    )

    # Add the new foreign key constraint with cascade delete
    op.create_foreign_key(
        'conversation_callbacks_conversation_id_fkey',
        'conversation_callbacks',
        'conversation_metadata',
        ['conversation_id'],
        ['conversation_id'],
        ondelete='CASCADE',
    )


def downgrade():
    # Drop the cascade delete foreign key constraint
    op.drop_constraint(
        'conversation_callbacks_conversation_id_fkey',
        'conversation_callbacks',
        type_='foreignkey',
    )

    # Recreate the original foreign key constraint without cascade delete
    op.create_foreign_key(
        'conversation_callbacks_conversation_id_fkey',
        'conversation_callbacks',
        'conversation_metadata',
        ['conversation_id'],
        ['conversation_id'],
    )
