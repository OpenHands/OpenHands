"""Add agent_profiles to org and active_agent_profile_id to org_member.

Agent Profiles (OpenHands/OpenHands#15044) are the named, reference-bearing
launch specs one level up from LLM profiles. They are stored at the
organization level exactly like ``org.llm_profiles``:

- ``org.agent_profiles`` is an EncryptedJSON column (stored as String). Agent
  profiles are reference-bearing and mostly secret-free, but the column is kept
  encrypted for parity with ``llm_profiles`` and because ``skills[].mcp_tools``
  env/headers ride in cleartext inside the encrypted blob (the column is the
  at-rest boundary). Envelope: ``{profiles: {<id>: AgentProfile}, active}``.
- ``org_member.active_agent_profile_id`` is the per-member pointer to the active
  profile id — the sole launch authority — mirroring the per-member
  ``_llm_api_key`` precedent.

Data migration: no backfill. Existing rows read back with ``agent_profiles =
NULL`` (treated as an empty collection) and ``active_agent_profile_id = NULL``
(falls back to the legacy active-LLM materialization). No profile is
auto-created: an org sees an empty list until someone with
``EDIT_ORG_SETTINGS`` explicitly saves one, so no downtime or follow-up
script is required.

Revision ID: 132
Revises: 131
Create Date: 2026-06-30 00:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = '132'
down_revision: Union[str, None] = '131'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('org', sa.Column('agent_profiles', sa.String(), nullable=True))
    op.add_column(
        'org_member',
        sa.Column('active_agent_profile_id', sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('org_member', 'active_agent_profile_id')
    op.drop_column('org', 'agent_profiles')
