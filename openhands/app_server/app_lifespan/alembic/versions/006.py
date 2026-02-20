"""Add environment_url column to conversation_metadata

Revision ID: 006
Revises: 005
Create Date: 2025-12-01 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '006'
down_revision: Union[str, Sequence[str], None] = '005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add environment_url column."""
    with op.batch_alter_table('conversation_metadata') as batch_op:
        batch_op.add_column(sa.Column('environment_url', sa.String(), nullable=True))


def downgrade() -> None:
    """Remove environment_url column."""
    with op.batch_alter_table('conversation_metadata') as batch_op:
        batch_op.drop_column('environment_url')
