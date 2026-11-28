"""Store automation outcome value facts.

Revision ID: 042
Revises: 041
Create Date: 2026-11-17
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "042"
down_revision: str = "041"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "factories",
        sa.Column(
            "outcome_keys", sa.JSON(), server_default=sa.text("'{}'"), nullable=False
        ),
    )
    columns = (
        "numeric_value",
        "text_value",
        "boolean_value",
        "datetime_value",
        "json_value",
    )
    types = ("number", "string", "boolean", "datetime", "json")
    typed_check = " OR ".join(
        f"(value_type = '{kind}' AND "
        + " AND ".join(
            f"{column} IS {'NOT ' if column == selected else ''}NULL"
            for column in columns
        )
        + ")"
        for kind, selected in zip(types, columns)
    )
    op.create_table(
        "automation_outcome_values",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("org_id", sa.Uuid(), nullable=False),
        sa.Column("factory_id", sa.Uuid(), nullable=False),
        sa.Column("automation_id", sa.Uuid(), nullable=True),
        sa.Column("run_id", sa.Uuid(), nullable=True),
        sa.Column("source", sa.String(20), nullable=False, server_default="agent"),
        sa.Column("key", sa.String(100), nullable=False),
        sa.Column("value_type", sa.String(20), nullable=False),
        sa.Column("numeric_value", sa.Float(), nullable=True),
        sa.Column("text_value", sa.Text(), nullable=True),
        sa.Column("boolean_value", sa.Boolean(), nullable=True),
        sa.Column("datetime_value", sa.DateTime(timezone=True), nullable=True),
        sa.Column("json_value", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "source IN ('agent', 'user', 'platform')", name="outcome_source"
        ),
        sa.CheckConstraint(
            "length(key) BETWEEN 1 AND 100 AND "
            "trim(key, 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            "0123456789_.-') = ''",
            name="outcome_key",
        ),
        sa.CheckConstraint(typed_check, name="outcome_typed_value"),
    )
    for name, fields in (
        ("ix_outcomes_factory_key_created", ["factory_id", "key", "created_at"]),
        ("ix_outcomes_org_factory_created", ["org_id", "factory_id", "created_at"]),
        (
            "ix_outcomes_factory_automation_created",
            ["factory_id", "automation_id", "created_at"],
        ),
        ("ix_outcomes_run_id", ["run_id"]),
    ):
        op.create_index(name, "automation_outcome_values", fields)


def downgrade() -> None:
    op.drop_table("automation_outcome_values")
    op.drop_column("factories", "outcome_keys")
