"""Add session_api_key_hash to v1_remote_sandbox table

Revision ID: 006
Revises: 005
Create Date: 2025-02-24 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '006'
down_revision: Union[str, None] = '005'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add session_api_key_hash column to v1_remote_sandbox table."""
    with op.batch_alter_table('v1_remote_sandbox') as batch_op:
        batch_op.add_column(
            sa.Column('session_api_key_hash', sa.String(), nullable=True)
        )
        batch_op.create_index(
            'ix_v1_remote_sandbox_session_api_key_hash',
            ['session_api_key_hash'],
            unique=False,
        )


def downgrade() -> None:
    """Remove session_api_key_hash column from v1_remote_sandbox table."""
    with op.batch_alter_table('v1_remote_sandbox') as batch_op:
        batch_op.drop_index('ix_v1_remote_sandbox_session_api_key_hash')
        batch_op.drop_column('session_api_key_hash')
