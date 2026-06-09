"""Add session_api_key_hash to v1_remote_sandbox table

Revision ID: 097
Revises: 096
Create Date: 2025-02-24 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '097'
down_revision: Union[str, None] = '096'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add session_api_key_hash column to v1_remote_sandbox table."""
    op.add_column(
        'v1_remote_sandbox',
        sa.Column('session_api_key_hash', sa.String(), nullable=True),
    )
    op.create_index(
        op.f('ix_v1_remote_sandbox_session_api_key_hash'),
        'v1_remote_sandbox',
        ['session_api_key_hash'],
        unique=False,
    )


def downgrade() -> None:
    """Remove session_api_key_hash column from v1_remote_sandbox table."""
    op.drop_index(
        op.f('ix_v1_remote_sandbox_session_api_key_hash'),
        table_name='v1_remote_sandbox',
    )
    op.drop_column('v1_remote_sandbox', 'session_api_key_hash')
