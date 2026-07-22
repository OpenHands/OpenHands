"""Ensure upgraded instances have an initial superadmin.

Revision ID: 138
Revises: 137
Create Date: 2026-07-21 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '138'
down_revision: Union[str, None] = '137'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    connection = op.get_bind()
    metadata = sa.MetaData()
    role = sa.Table('role', metadata, autoload_with=connection)
    user = sa.Table('user', metadata, autoload_with=connection)

    admin_role_id = sa.select(role.c.id).where(role.c.name == 'admin').scalar_subquery()
    has_superadmin = sa.exists(
        sa.select(user.c.id).where(user.c.role_id == admin_role_id)
    )
    oldest_user_id = (
        sa.select(user.c.id)
        .order_by(
            sa.func.coalesce(user.c.first_login_at, user.c.accepted_tos)
            .asc()
            .nulls_last(),
            user.c.id.asc(),
        )
        .limit(1)
        .scalar_subquery()
    )

    connection.execute(
        sa.update(user)
        .where(user.c.id == oldest_user_id, ~has_superadmin)
        .values(role_id=admin_role_id)
    )


def downgrade() -> None:
    pass
