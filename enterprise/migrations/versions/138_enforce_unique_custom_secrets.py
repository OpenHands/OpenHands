"""Enforce unique custom-secret names per user and organization."""

import sqlalchemy as sa
from alembic import op

revision = '138'
down_revision = '137'
branch_labels = None
depends_on = None

_INDEX_NAME = 'uq_custom_secrets_user_org_secret_name'


def upgrade() -> None:
    op.execute(
        sa.text(
            """
            DELETE FROM custom_secrets
            WHERE id NOT IN (
                SELECT MAX(id)
                FROM custom_secrets
                GROUP BY keycloak_user_id, org_id, secret_name
            )
            """
        )
    )
    op.create_index(
        _INDEX_NAME,
        'custom_secrets',
        ['keycloak_user_id', 'org_id', 'secret_name'],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index(_INDEX_NAME, table_name='custom_secrets')
