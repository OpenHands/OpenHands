"""Add SETTING_UP_SKILLS to appconversationstarttaskstatus enum

Revision ID: 082
Revises: 081
Create Date: 2025-11-19 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision: str = '082'
down_revision: Union[str, Sequence[str], None] = '081'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add SETTING_UP_SKILLS enum value to appconversationstarttaskstatus."""
    # Check if the enum value already exists before adding it
    # This handles the case where the enum was created with the value already included
    connection = op.get_bind()
    result = connection.execute(
        text(
            "SELECT 1 FROM pg_enum WHERE enumlabel = 'SETTING_UP_SKILLS' "
            "AND enumtypid = (SELECT oid FROM pg_type WHERE typname = 'appconversationstarttaskstatus')"
        )
    )

    if not result.fetchone():
        # Add the new enum value only if it doesn't already exist
        op.execute(
            "ALTER TYPE appconversationstarttaskstatus ADD VALUE 'SETTING_UP_SKILLS'"
        )


def downgrade() -> None:
    """Remove SETTING_UP_SKILLS enum value from appconversationstarttaskstatus.

    Note: PostgreSQL doesn't support removing enum values directly.
    This would require recreating the enum type and updating all references.
    For safety, this downgrade is not implemented.
    """
    # PostgreSQL doesn't support removing enum values directly
    # This would require a complex migration to recreate the enum
    # For now, we'll leave this as a no-op since removing enum values
    # is rarely needed and can be dangerous
    pass
