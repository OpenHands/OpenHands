"""Add acp_server column to conversation_metadata table.

Revision ID: 010
Revises: 009
Create Date: 2026-04-29 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    existing = [
        col["name"] for col in sa.inspect(bind).get_columns("conversation_metadata")
    ]
    if "acp_server" not in existing:
        op.add_column(
            "conversation_metadata", sa.Column("acp_server", sa.String, nullable=True)
        )


def downgrade() -> None:
    bind = op.get_bind()
    existing = [
        col["name"] for col in sa.inspect(bind).get_columns("conversation_metadata")
    ]
    if "acp_server" in existing:
        op.drop_column("conversation_metadata", "acp_server")
