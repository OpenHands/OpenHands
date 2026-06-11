"""Add is_default flag to org for default-org bootstrap keying

The OHE default-org bootstrap previously located its org by configured name,
so renaming the org (in KOTS or in-app) silently forked a second org. The
bootstrap now marks the org it creates (or adopts) with is_default and looks
it up by that flag, making the org freely renameable.

A partial unique index guarantees at most one default org per install.

Revision ID: 119
Revises: 118
Create Date: 2026-06-10
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '119'
down_revision: Union[str, None] = '118'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'org',
        sa.Column(
            'is_default',
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.create_index(
        'uq_org_is_default',
        'org',
        ['is_default'],
        unique=True,
        postgresql_where=sa.text('is_default'),
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index('uq_org_is_default', table_name='org', if_exists=True)
    op.drop_column('org', 'is_default')
