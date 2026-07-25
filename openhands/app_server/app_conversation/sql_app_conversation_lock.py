from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession


async def lock_app_conversation(
    session: AsyncSession, conversation_id: UUID | str
) -> None:
    conversation_id = UUID(str(conversation_id))
    if session.get_bind().dialect.name == 'postgresql':
        await session.execute(
            select(
                func.pg_advisory_xact_lock(
                    func.hashtext(f'app-conversation:{conversation_id}')
                )
            )
        )
