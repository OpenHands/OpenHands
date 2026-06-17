"""Remove max concurrent sandbox limit columns.

Revision ID: 124
Revises: 123
Create Date: 2026-06-17 00:00:00.000000
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '124'
down_revision: str | None = '123'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table('org_member') as batch_op:
        batch_op.drop_column('max_concurrent_sandboxes_override')

    with op.batch_alter_table('org') as batch_op:
        batch_op.drop_column('max_concurrent_sandboxes')


def downgrade() -> None:
    with op.batch_alter_table('org') as batch_op:
        batch_op.add_column(
            sa.Column(
                'max_concurrent_sandboxes',
                sa.Integer(),
                nullable=False,
                server_default='10',
            )
        )

    with op.batch_alter_table('org_member') as batch_op:
        batch_op.add_column(
            sa.Column(
                'max_concurrent_sandboxes_override',
                sa.Integer(),
                nullable=True,
            )
        )

    op.execute("""
        UPDATE org
        SET max_concurrent_sandboxes = 3
        WHERE id IN (SELECT id FROM "user")
    """)
