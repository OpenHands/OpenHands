"""Tests for the conversation_cost_events foreign key cascade behavior.

The ``conversation_cost_events`` table records per-event cost deltas whose
running total is already mirrored on ``conversation_metadata.accumulated_cost``.
Deleting a conversation must therefore also remove its cost-event rows so
that ``delete_app_conversation_info`` does not fail with a
``ForeignKeyViolationError``.

These tests guard two invariants:

1. The SQLAlchemy model declares the FK with ``ondelete='CASCADE'`` so
   fresh-schema creation (tests, dev) matches the production migration.
2. Deleting a conversation row actually removes its cost-event rows on a
   backend that enforces foreign keys (SQLite with ``PRAGMA foreign_keys=ON``).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import AsyncGenerator
from uuid import UUID, uuid4

import pytest
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from openhands.app_server.app_conversation.sql_app_conversation_info_service import (
    SQLAppConversationInfoService,
    StoredConversationCostEvent,
    StoredConversationMetadata,
)
from openhands.app_server.user.specifiy_user_context import SpecifyUserContext


@pytest.fixture
async def engine() -> AsyncGenerator[AsyncEngine, None]:
    """Async SQLite engine with FK enforcement enabled.

    SQLite does not enforce foreign-key constraints unless each connection
    runs ``PRAGMA foreign_keys=ON``; the engine ``connect`` event listener
    applies it automatically.
    """

    engine = create_async_engine(
        'sqlite+aiosqlite:///:memory:',
        poolclass=StaticPool,
        connect_args={'check_same_thread': False},
    )

    @event.listens_for(engine.sync_engine, 'connect')
    def _enable_sqlite_fk(dbapi_connection, _connection_record):  # pragma: no cover
        cursor = dbapi_connection.cursor()
        cursor.execute('PRAGMA foreign_keys=ON')
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(StoredConversationMetadata.metadata.create_all)

    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def session(engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    session_maker = async_sessionmaker(engine, expire_on_commit=False)
    async with session_maker() as db_session:
        yield db_session


def test_cost_event_fk_declares_cascade():
    """The model FK must declare ``ondelete='CASCADE'``.

    Without this, any fresh database (tests, dev, on-prem deploys that
    re-create the schema) ends up with the same broken constraint the
    production migration is fixing.
    """
    conversation_id_column = StoredConversationCostEvent.__table__.c.conversation_id
    foreign_keys = list(conversation_id_column.foreign_keys)
    assert len(foreign_keys) == 1
    fk = foreign_keys[0]
    assert fk.column.table.name == 'conversation_metadata'
    assert fk.ondelete == 'CASCADE'


@pytest.mark.asyncio
async def test_delete_conversation_cascades_cost_events(
    engine: AsyncEngine, session: AsyncSession
):
    """Deleting a conversation must remove its cost-event rows.

    Reproduces the production bug where deleting from
    ``conversation_metadata`` fails with ``ForeignKeyViolationError`` because
    ``conversation_cost_events`` still references the row.
    """
    conversation_id = str(uuid4())
    other_conversation_id = str(uuid4())

    session.add(
        StoredConversationMetadata(
            conversation_id=conversation_id,
            created_at=datetime.now(timezone.utc),
            last_updated_at=datetime.now(timezone.utc),
        )
    )
    session.add(
        StoredConversationMetadata(
            conversation_id=other_conversation_id,
            created_at=datetime.now(timezone.utc),
            last_updated_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()

    now = datetime.now(timezone.utc)
    session.add_all(
        [
            StoredConversationCostEvent(
                conversation_id=conversation_id, cost_delta=0.10, occurred_at=now
            ),
            StoredConversationCostEvent(
                conversation_id=conversation_id, cost_delta=0.25, occurred_at=now
            ),
            StoredConversationCostEvent(
                conversation_id=other_conversation_id,
                cost_delta=0.99,
                occurred_at=now,
            ),
        ]
    )
    await session.commit()

    service = SQLAppConversationInfoService(
        db_session=session, user_context=SpecifyUserContext(user_id=None)
    )
    deleted = await service.delete_app_conversation_info(UUID(conversation_id))
    assert deleted is True

    remaining_metadata = (
        (await session.execute(select(StoredConversationMetadata.conversation_id)))
        .scalars()
        .all()
    )
    assert remaining_metadata == [other_conversation_id]

    remaining_cost_events = (
        (await session.execute(select(StoredConversationCostEvent.conversation_id)))
        .scalars()
        .all()
    )
    assert remaining_cost_events == [other_conversation_id]
