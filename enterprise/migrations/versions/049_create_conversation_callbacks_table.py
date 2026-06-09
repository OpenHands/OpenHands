"""Create conversation callbacks table

Revision ID: 049
Revises: 048
Create Date: 2025-06-19

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '049'
down_revision = '048'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'conversation_callbacks',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('conversation_id', sa.String(), nullable=False),
        sa.Column(
            'status',
            sa.Enum('ACTIVE', 'COMPLETED', 'ERROR', name='callbackstatus'),
            nullable=False,
        ),
        sa.Column('processor_type', sa.String(), nullable=False),
        sa.Column('processor_json', sa.Text(), nullable=False),
        sa.Column(
            'created_at',
            sa.DateTime(),
            server_default=sa.text('CURRENT_TIMESTAMP'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(),
            server_default=sa.text('CURRENT_TIMESTAMP'),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(
            ['conversation_id'],
            ['conversation_metadata.conversation_id'],
        ),
    )
    op.create_index(
        op.f('ix_conversation_callbacks_conversation_id'),
        'conversation_callbacks',
        ['conversation_id'],
        unique=False,
    )


def downgrade():
    op.drop_index(
        op.f('ix_conversation_callbacks_conversation_id'),
        table_name='conversation_callbacks',
    )
    op.drop_table('conversation_callbacks')
    op.execute('DROP TYPE callbackstatus')
