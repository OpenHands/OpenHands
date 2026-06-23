"""Add 'empty' role used as a carrier for super-only permissions.

The ``empty`` role intentionally has no org-scoped permissions. It exists
so that a user can be granted a super role (via ``user.role_id``) whose
effective permissions come entirely from
``SUPER_ROLE_ADDITIONAL_PERMISSIONS`` -- for example, granting only the
``create_organization`` permission to a ``superempty`` user.

Revision ID: 125
Revises: 124
Create Date: 2026-06-23 00:00:00.000000
"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '125'
down_revision: str | None = '124'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Rank is set higher than ``member`` (1000) so the role sorts last in
    # rank-ordered listings; it carries no org-scoped permissions.
    op.execute(
        sa.text(
            "INSERT INTO role (name, rank) VALUES ('empty', 2000) "
            'ON CONFLICT (name) DO NOTHING'
        )
    )


def downgrade() -> None:
    op.execute(sa.text("DELETE FROM role WHERE name = 'empty'"))
