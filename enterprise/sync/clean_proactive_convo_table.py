import asyncio  # noqa: I001

# This must be before the import of storage
# to set up logging and prevent alembic from
# running its mouth.
from openhands.core.logger import openhands_logger

from storage.proactive_conversation_store import (
    ProactiveConversationStore,
)

OLDER_THAN = 30  # 30 minutes


async def main():
    openhands_logger.info('clean_proactive_convo_table')
    convo_store = ProactiveConversationStore()
    await convo_store.clean_old_convos(older_than_minutes=OLDER_THAN)


if __name__ == '__main__':
    asyncio.run(main())
