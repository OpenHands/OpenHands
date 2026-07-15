"""Enforce custom secret uniqueness."""

import sqlalchemy as sa
from alembic import op

revision = '138'
down_revision = '137'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        sa.text(
            """
            DELETE FROM custom_secrets
            WHERE id IN (
                SELECT id
                FROM (
                    SELECT id, ROW_NUMBER() OVER (
                        PARTITION BY keycloak_user_id, org_id, secret_name
                        ORDER BY id DESC
                    ) AS duplicate_rank
                    FROM custom_secrets
                ) ranked
                WHERE duplicate_rank > 1
            )
            """
        )
    )
    op.create_unique_constraint(
        'uq_custom_secrets_user_org_name',
        'custom_secrets',
        ['keycloak_user_id', 'org_id', 'secret_name'],
    )


def downgrade() -> None:
    op.drop_constraint(
        'uq_custom_secrets_user_org_name',
        'custom_secrets',
        type_='unique',
    )
