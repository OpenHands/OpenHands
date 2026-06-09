"""create proactive conversation starters commits table

Revision ID: 034
Revises: 033
Create Date: 2024-03-11 23:39:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '034'
down_revision: Union[str, None] = '033'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'proactive_conversation_table',
        sa.Column('id', sa.Integer(), sa.Identity(), nullable=False, primary_key=True),
        sa.Column('repo_id', sa.String(), nullable=False),
        sa.Column('pr_number', sa.Integer(), nullable=False),
        sa.Column('workflow_runs', sa.JSON(), nullable=False),
        sa.Column('commit', sa.String(), nullable=False),
        sa.Column(
            'conversation_starter_sent', sa.Boolean(), nullable=False, default=False
        ),
        sa.Column('last_updated_at', sa.DateTime(), nullable=False),
    )

    op.create_index(
        'ix_proactive_conversation_repo_pr',  # Index name
        'proactive_conversation_table',  # Table name
        ['repo_id', 'pr_number'],  # Columns to index
        unique=False,  # Set to True if you want a unique index
    )


def downgrade() -> None:
    op.drop_table('proactive_conversation_table')
    op.drop_index(
        'ix_proactive_conversation_repo_pr', table_name='proactive_conversation_table'
    )
