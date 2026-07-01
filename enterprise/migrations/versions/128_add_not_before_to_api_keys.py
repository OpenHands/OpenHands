"""Add not_before column to api_keys table.

Adds an optional ``not_before`` timestamp representing the earliest moment at
which an API key becomes valid. Combined with the existing ``expires_at``
column, this lets users define an explicit active window for a key
(``not_before <= now < expires_at``). Both bounds remain independently
optional and additive: existing rows get ``NULL`` (immediately valid) so
behaviour is unchanged for every key created before this migration.

Revision ID: 128
Revises: 127
Create Date: 2026-06-05 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '128'
down_revision: Union[str, None] = '127'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table('api_keys') as batch_op:
        batch_op.add_column(sa.Column('not_before', sa.DateTime(), nullable=True))
        batch_op.create_index(
            'ix_api_keys_not_before',
            'api_keys',
            ['not_before'],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table('api_keys') as batch_op:
        batch_op.drop_index('ix_api_keys_not_before')
        batch_op.drop_column('not_before')
