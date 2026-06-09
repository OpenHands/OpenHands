"""rename user_secrets table to custom_secrets

Revision ID: 079
Revises: 078
Create Date: 2025-10-27 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '079'
down_revision: Union[str, None] = '078'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Rename the table from user_secrets to custom_secrets
    op.rename_table('user_secrets', 'custom_secrets')

    # Rename the index to match the new table name
    op.drop_index('idx_user_secrets_keycloak_user_id', 'custom_secrets')
    op.create_index(
        'idx_custom_secrets_keycloak_user_id', 'custom_secrets', ['keycloak_user_id']
    )


def downgrade() -> None:
    # Rename the index back to the original name
    op.drop_index('idx_custom_secrets_keycloak_user_id', 'custom_secrets')
    op.create_index(
        'idx_user_secrets_keycloak_user_id', 'custom_secrets', ['keycloak_user_id']
    )

    # Rename the table back from custom_secrets to user_secrets
    op.rename_table('custom_secrets', 'user_secrets')
