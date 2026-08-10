"""Initial engagement-manager schema.

Revision ID: 001
Revises:
Create Date: 2026-08-10
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
    op.create_table(
        "engagements",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("client_name", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "status", sa.String(32), nullable=False, server_default="draft"
        ),
        sa.Column("scope_authorized_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("scope_document_url", sa.Text(), nullable=True),
        sa.Column(
            "autonomy_mode",
            sa.String(32),
            nullable=False,
            server_default="semi_autonomous",
        ),
        sa.Column(
            "runtime_profile", sa.String(32), nullable=False, server_default="web"
        ),
        sa.Column(
            "sandbox_status", sa.String(32), server_default="stopped"
        ),
        sa.Column("sandbox_compose_project", sa.Text(), nullable=True),
        sa.Column("defectdojo_engagement_id", sa.Integer(), nullable=True),
        sa.Column("created_by", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('draft','active','paused','completed','archived')",
            name="engagements_status_check",
        ),
        sa.CheckConstraint(
            "autonomy_mode IN ('manual','semi_autonomous','autonomous')",
            name="engagements_autonomy_check",
        ),
        sa.CheckConstraint(
            "runtime_profile IN ('web','network','mobile','sast')",
            name="engagements_runtime_check",
        ),
    )
    op.create_table(
        "scope_rules",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(
            "engagement_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("engagements.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("rule_type", sa.String(16), nullable=False),
        sa.Column("target_type", sa.String(16), nullable=False),
        sa.Column("target_value", sa.Text(), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )
    op.create_index("idx_engagements_status", "engagements", ["status"])
    op.create_index(
        "idx_scope_rules_engagement_id", "scope_rules", ["engagement_id"]
    )


def downgrade() -> None:
    op.drop_index("idx_scope_rules_engagement_id", table_name="scope_rules")
    op.drop_index("idx_engagements_status", table_name="engagements")
    op.drop_table("scope_rules")
    op.drop_table("engagements")
