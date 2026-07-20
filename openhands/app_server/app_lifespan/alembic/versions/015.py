"""Add sandbox teardown session keys."""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '015'
down_revision: str | None = '014'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table('v1_remote_sandbox') as batch_op:
        batch_op.add_column(
            sa.Column('teardown_session_api_key_hash', sa.String(), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                'teardown_session_api_key_expires_at',
                sa.DateTime(timezone=True),
                nullable=True,
            )
        )
        batch_op.create_index(
            'ix_v1_remote_sandbox_teardown_session_api_key_hash',
            ['teardown_session_api_key_hash'],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table('v1_remote_sandbox') as batch_op:
        batch_op.drop_index('ix_v1_remote_sandbox_teardown_session_api_key_hash')
        batch_op.drop_column('teardown_session_api_key_expires_at')
        batch_op.drop_column('teardown_session_api_key_hash')
