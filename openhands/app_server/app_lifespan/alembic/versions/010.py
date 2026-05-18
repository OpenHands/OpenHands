"""Add v1_boxd_sandbox table for the boxd sandbox backend.

Revision ID: 010
Revises: 009
Create Date: 2026-05-18 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "010"
down_revision: Union[str, None] = "009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "v1_boxd_sandbox",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("created_by_user_id", sa.String(), nullable=True),
        sa.Column("sandbox_spec_id", sa.String(), nullable=False),
        sa.Column("session_api_key_hash", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_v1_boxd_sandbox_created_at"),
        "v1_boxd_sandbox",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_v1_boxd_sandbox_created_by_user_id"),
        "v1_boxd_sandbox",
        ["created_by_user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_v1_boxd_sandbox_sandbox_spec_id"),
        "v1_boxd_sandbox",
        ["sandbox_spec_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_v1_boxd_sandbox_session_api_key_hash"),
        "v1_boxd_sandbox",
        ["session_api_key_hash"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_v1_boxd_sandbox_session_api_key_hash"),
        table_name="v1_boxd_sandbox",
    )
    op.drop_index(
        op.f("ix_v1_boxd_sandbox_sandbox_spec_id"),
        table_name="v1_boxd_sandbox",
    )
    op.drop_index(
        op.f("ix_v1_boxd_sandbox_created_by_user_id"),
        table_name="v1_boxd_sandbox",
    )
    op.drop_index(
        op.f("ix_v1_boxd_sandbox_created_at"),
        table_name="v1_boxd_sandbox",
    )
    op.drop_table("v1_boxd_sandbox")
