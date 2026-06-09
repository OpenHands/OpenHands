"""bitbucket_dc_webhook.last_synced TIMESTAMP WITHOUT TIME ZONE -> WITH TIME ZONE

Revision ID: 113
Revises: 112
Create Date: 2026-05-19 00:00:00.000000

The original migration (112) created ``last_synced`` as
``TIMESTAMP WITHOUT TIME ZONE``, which meant the route layer had to
strip ``tzinfo`` from every ``datetime.now(timezone.utc)`` value before
asyncpg would accept it. Converting to ``TIMESTAMP WITH TIME ZONE``
lets the model store offset-aware datetimes directly and removes the
``.replace(tzinfo=None)`` workarounds in ``bitbucket_dc_webhook_store``.

The conversion is in-place — Postgres reads existing naive values as
UTC during the column-type change, which is what they always
semantically were.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '113'
down_revision: Union[str, None] = '112'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        'bitbucket_dc_webhook',
        'last_synced',
        type_=sa.DateTime(timezone=True),
        existing_type=sa.DateTime(timezone=False),
        existing_server_default=sa.text('CURRENT_TIMESTAMP'),
        existing_nullable=True,
        postgresql_using="last_synced AT TIME ZONE 'UTC'",
    )


def downgrade() -> None:
    op.alter_column(
        'bitbucket_dc_webhook',
        'last_synced',
        type_=sa.DateTime(timezone=False),
        existing_type=sa.DateTime(timezone=True),
        existing_server_default=sa.text('CURRENT_TIMESTAMP'),
        existing_nullable=True,
        postgresql_using="last_synced AT TIME ZONE 'UTC'",
    )
