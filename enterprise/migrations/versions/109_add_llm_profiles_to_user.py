"""Add llm_profiles column to user table.

The Settings model exposes ``llm_profiles`` (saved LLM configurations plus
the active profile name), but the SaaS path persists a flattened Settings
dump onto the User/Org rows. Without a column here the field is silently
dropped on store() and always defaults to empty on load(), so saved
profiles disappear after any settings update or page refresh.

The column is plain ``String`` because the ORM-level ``EncryptedJSON``
TypeDecorator stores JSON-serialized profiles as a JWE-encrypted string —
profiles can carry per-profile ``api_key`` values, so the at-rest
representation must match the existing org/member encrypted-secret pattern.

Revision ID: 109
Revises: 108
Create Date: 2026-04-28
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '109'
down_revision: Union[str, None] = '108'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('user', sa.Column('llm_profiles', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('user', 'llm_profiles')
