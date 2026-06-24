"""Add dependency_repos column to conversation_metadata

Revision ID: 013
Revises: 012
Create Date: 2026-06-24 00:00:00.000000
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '013'
down_revision: str | None = '012'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add dependency_repos JSON column to conversation_metadata.

    Stores a list of dependency repository specs (name, repo, ref)
    that the agent can work with alongside the primary repository.
    """
    with op.batch_alter_table('conversation_metadata') as batch_op:
        batch_op.add_column(
            sa.Column(
                'dependency_repos',
                sa.JSON(),
                nullable=True,
            )
        )


def downgrade() -> None:
    with op.batch_alter_table('conversation_metadata') as batch_op:
        batch_op.drop_column('dependency_repos')
