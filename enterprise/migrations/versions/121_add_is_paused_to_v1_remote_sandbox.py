"""Add is_paused to v1_remote_sandbox.

The OpenHands app-server RemoteSandboxService stores pause state on
v1_remote_sandbox and queries it during concurrency-limit checks. Enterprise
migrations create the table separately from the OSS app-lifespan migrations, so
SaaS deployments need this column in the enterprise migration chain as well.

Revision ID: 121
Revises: 120
Create Date: 2026-06-15
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '121'
down_revision: Union[str, None] = '120'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'v1_remote_sandbox',
        sa.Column(
            'is_paused',
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )


def downgrade() -> None:
    op.drop_column('v1_remote_sandbox', 'is_paused')
