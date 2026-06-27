"""Add llm_profiles column to org table.

LLM profiles are stored at the organization level to support both personal
workspaces (where the owner manages their own profiles) and team organizations
(where admins manage profiles that members can activate).

The column uses EncryptedJSON (stored as String) because profiles can contain
per-profile API keys that must be encrypted at rest.

Data migration: no backfill. Existing orgs read back with ``llm_profiles =
NULL`` and ``_load_profiles`` treats that as an empty ``LLMProfiles``. The
first save through the ``/api/organizations/{org_id}/profiles`` endpoints
populates the column lazily, so no downtime or follow-up script is required.

Revision ID: 116
Revises: 115
Create Date: 2025-05-13
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '116'
down_revision: Union[str, None] = '115'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('org', sa.Column('llm_profiles', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('org', 'llm_profiles')
