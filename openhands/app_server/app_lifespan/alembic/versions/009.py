"""Add critic result columns to conversation_metadata table

Revision ID: 009
Revises: 008
Create Date: 2026-04-14 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '009'
down_revision: Union[str, None] = '008'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add critic evaluation result columns.

    Populated by the FinishCriticCallbackProcessor once a V1 conversation
    reaches a terminal execution status. Columns are nullable for
    backwards compatibility with existing rows.
    """
    op.add_column(
        'conversation_metadata',
        sa.Column('critic_score', sa.Float, nullable=True),
    )
    op.add_column(
        'conversation_metadata',
        sa.Column('critic_message', sa.String, nullable=True),
    )
    op.add_column(
        'conversation_metadata',
        sa.Column('critic_evaluated_at', sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    """Remove critic evaluation result columns."""
    op.drop_column('conversation_metadata', 'critic_evaluated_at')
    op.drop_column('conversation_metadata', 'critic_message')
    op.drop_column('conversation_metadata', 'critic_score')
