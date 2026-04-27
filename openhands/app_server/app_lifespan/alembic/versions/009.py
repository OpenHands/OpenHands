"""Add llm_base_url column to conversation_metadata table.

Revision ID: 009
Revises: 008
Create Date: 2026-04-22 00:00:00.000000

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
    """Add llm_base_url for conversation-specific LLM endpoint persistence."""
    op.add_column(
        'conversation_metadata',
        sa.Column('llm_base_url', sa.String(), nullable=True),
    )


def downgrade() -> None:
    """Remove llm_base_url column from conversation_metadata."""
    op.drop_column('conversation_metadata', 'llm_base_url')
