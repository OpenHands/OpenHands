"""widen conversation_metadata token counters to BigInteger

Cumulative token usage on long-running or expensive conversations can exceed the
PostgreSQL Integer (int32) maximum of 2,147,483,647. Writing such a value raises
``asyncpg.exceptions.DataError: value out of int32 range``, which fails the flush
and leaves the session unusable, producing a sustained stream of 500s for that
conversation. Widen the token counters to BigInteger (int64).

Revision ID: 005
Revises: 004
Create Date: 2026-09-30 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '005'
down_revision: Union[str, None] = '004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


TOKEN_COLUMNS = (
    'prompt_tokens',
    'completion_tokens',
    'total_tokens',
    'cache_read_tokens',
    'cache_write_tokens',
    'reasoning_tokens',
    'context_window',
    'per_turn_token',
)


def upgrade() -> None:
    """Upgrade schema."""
    for column in TOKEN_COLUMNS:
        op.alter_column(
            'conversation_metadata',
            column,
            existing_type=sa.Integer(),
            type_=sa.BigInteger(),
            existing_nullable=True,
        )


def downgrade() -> None:
    """Downgrade schema."""
    for column in TOKEN_COLUMNS:
        op.alter_column(
            'conversation_metadata',
            column,
            existing_type=sa.BigInteger(),
            type_=sa.Integer(),
            existing_nullable=True,
        )
