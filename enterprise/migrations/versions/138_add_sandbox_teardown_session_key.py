"""Add sandbox teardown session keys."""

import sqlalchemy as sa
from alembic import op

revision = '138'
down_revision = '137'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'v1_remote_sandbox',
        sa.Column('teardown_session_api_key_hash', sa.String(), nullable=True),
    )
    op.add_column(
        'v1_remote_sandbox',
        sa.Column(
            'teardown_session_api_key_expires_at',
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )
    op.create_index(
        op.f('ix_v1_remote_sandbox_teardown_session_api_key_hash'),
        'v1_remote_sandbox',
        ['teardown_session_api_key_hash'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f('ix_v1_remote_sandbox_teardown_session_api_key_hash'),
        table_name='v1_remote_sandbox',
    )
    op.drop_column(
        'v1_remote_sandbox',
        'teardown_session_api_key_expires_at',
    )
    op.drop_column(
        'v1_remote_sandbox',
        'teardown_session_api_key_hash',
    )
