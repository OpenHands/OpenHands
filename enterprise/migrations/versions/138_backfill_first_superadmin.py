"""Backfill a superadmin when upgraded instances have users but none.

Revision ID: 138
Revises: 137
Create Date: 2026-07-20

Upgrades that introduced ``user.role_id`` left existing tenants with users
but zero superadmins (#15327). Promote the earliest user (by first_login_at,
then email, then id) when the admin role exists and nobody holds it.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "138"
down_revision = "137"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    admin_role_id = conn.execute(
        sa.text("SELECT id FROM role WHERE name = 'admin' LIMIT 1")
    ).scalar()
    if admin_role_id is None:
        return

    superadmin_count = conn.execute(
        sa.text('SELECT COUNT(*) FROM "user" WHERE role_id = :rid'),
        {"rid": admin_role_id},
    ).scalar()
    if superadmin_count and superadmin_count > 0:
        return

    user_count = conn.execute(sa.text('SELECT COUNT(*) FROM "user"')).scalar()
    if not user_count:
        return

    # Prefer the first-ever login; fall back to email/id for deterministic pick.
    target = conn.execute(
        sa.text(
            'SELECT id FROM "user" '
            "ORDER BY first_login_at ASC NULLS LAST, email ASC NULLS LAST, id ASC "
            "LIMIT 1"
        )
    ).scalar()
    if target is None:
        return

    conn.execute(
        sa.text('UPDATE "user" SET role_id = :rid WHERE id = :uid'),
        {"rid": admin_role_id, "uid": target},
    )


def downgrade() -> None:
    # Non-destructive: do not strip superadmin grants on downgrade.
    pass
