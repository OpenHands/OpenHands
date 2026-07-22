"""Add max_concurrent_sandboxes to org and org_member tables.

Adds per-org default and per-user override for concurrent sandbox limits.
- org.max_concurrent_sandboxes: org-wide default
  - Personal orgs (org.id == user.id): default 3 (set at user creation time)
  - Commercial orgs: default 10 (set at org creation time, server_default used for existing)
- org_member.max_concurrent_sandboxes_override: per-user override (NULL = use org default)

Note: Hardcoded values (3, 10) are intentional in migrations - they represent the
state at migration time and should not change if constants are updated later.
See server.constants for current values: DEFAULT_PERSONAL_ORG_CONCURRENT_SANDBOXES,
DEFAULT_COMMERCIAL_ORG_CONCURRENT_SANDBOXES.

Revision ID: 120
Revises: 119
Create Date: 2026-04-29
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '120'
down_revision: Union[str, None] = '119'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add max_concurrent_sandboxes to org table with default of 10 for commercial orgs
    # Personal orgs get limit of 3 set explicitly at user creation time
    op.add_column(
        'org',
        sa.Column(
            'max_concurrent_sandboxes',
            sa.Integer(),
            nullable=False,
            server_default='10',
        ),
    )

    # Add max_concurrent_sandboxes_override to org_member table (NULL = use org default)
    op.add_column(
        'org_member',
        sa.Column(
            'max_concurrent_sandboxes_override',
            sa.Integer(),
            nullable=True,
        ),
    )

    # Set personal orgs (where org.id exists in user table) to have a limit of 3
    op.execute("""
        UPDATE org
        SET max_concurrent_sandboxes = 3
        WHERE id IN (SELECT id FROM "user")
    """)


def downgrade() -> None:
    op.drop_column('org_member', 'max_concurrent_sandboxes_override')
    op.drop_column('org', 'max_concurrent_sandboxes')
