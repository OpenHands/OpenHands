"""Add org_id column to device_codes table.

Revision ID: 128
Revises: 127
Create Date: 2026-06-05

"""

from typing import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = '128'
down_revision: str | None = '127'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'device_codes',
        sa.Column(
            'org_id',
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )
    op.create_foreign_key(
        'fk_device_codes_org_id_org',
        'device_codes',
        'org',
        ['org_id'],
        ['id'],
    )


def downgrade() -> None:
    op.drop_constraint('fk_device_codes_org_id_org', 'device_codes', type_='foreignkey')
    op.drop_column('device_codes', 'org_id')
