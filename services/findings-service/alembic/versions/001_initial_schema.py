"""Initial findings schema.

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
        "findings",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("engagement_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_tool", sa.String(64), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("severity", sa.String(16), nullable=False),
        sa.Column("asset", sa.Text(), nullable=True),
        sa.Column("endpoint", sa.Text(), nullable=True),
        sa.Column("evidence", postgresql.JSONB(), nullable=True),
        sa.Column(
            "status", sa.String(32), nullable=False, server_default="new"
        ),
        sa.Column("dedupe_hash", sa.String(64), nullable=True),
        sa.Column("fp_reason", sa.Text(), nullable=True),
        sa.Column("triaged_by", sa.String(256), nullable=True),
        sa.Column("triaged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("defectdojo_id", sa.Integer(), nullable=True),
        sa.Column("defectdojo_synced_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cvss_score", sa.Numeric(4, 1), nullable=True),
        sa.Column("cve_ids", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("tags", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("created_by", sa.String(256), nullable=False),
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
            "status IN ('new','triaging','confirmed','false_positive',"
            "'duplicate','risk_accepted')",
            name="findings_status_check",
        ),
        sa.CheckConstraint(
            "severity IN ('critical','high','medium','low','info')",
            name="findings_severity_check",
        ),
    )
    op.create_index("idx_findings_engagement_id", "findings", ["engagement_id"])
    op.create_index("idx_findings_created_by", "findings", ["created_by"])
    op.create_index("idx_findings_status", "findings", ["status"])
    op.create_index("idx_findings_severity", "findings", ["severity"])
    op.execute(
        "CREATE UNIQUE INDEX idx_findings_dedupe ON findings(dedupe_hash) "
        "WHERE dedupe_hash IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_findings_dedupe")
    op.drop_index("idx_findings_severity", table_name="findings")
    op.drop_index("idx_findings_status", table_name="findings")
    op.drop_index("idx_findings_created_by", table_name="findings")
    op.drop_index("idx_findings_engagement_id", table_name="findings")
    op.drop_table("findings")
