"""Unit tests for BitbucketDCWebhookStore."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool
from storage.base import Base
from storage.bitbucket_dc_webhook import BitbucketDCWebhook
from storage.bitbucket_dc_webhook_store import BitbucketDCWebhookStore


@pytest.fixture(scope='function')
def event_loop():
    import asyncio

    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope='function')
async def async_engine(event_loop):
    engine = create_async_engine(
        'sqlite+aiosqlite:///:memory:',
        poolclass=StaticPool,
        connect_args={'check_same_thread': False},
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture(scope='function')
async def async_session_maker(async_engine):
    return async_sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def webhook_store(async_session_maker):
    import storage.bitbucket_dc_webhook_store as store_module

    store_module.a_session_maker = async_session_maker
    return BitbucketDCWebhookStore()


@pytest.fixture
async def sample_webhook(async_session_maker):
    async with async_session_maker() as session:
        webhook = BitbucketDCWebhook(
            project_key='PROJ',
            repo_slug='myrepo',
            user_id='kc-installer',
            webhook_id='42',
            webhook_secret='shared-secret',
        )
        session.add(webhook)
        await session.commit()
        await session.refresh(webhook)
    return webhook


@pytest.mark.asyncio
async def test_get_webhook_secret_returns_secret_for_matching_repo(
    webhook_store, sample_webhook
):
    secret = await webhook_store.get_webhook_secret(
        project_key='PROJ', repo_slug='myrepo'
    )
    assert secret == 'shared-secret'


@pytest.mark.asyncio
async def test_get_webhook_secret_returns_none_when_repo_not_registered(webhook_store):
    secret = await webhook_store.get_webhook_secret(
        project_key='OTHER', repo_slug='nope'
    )
    assert secret is None


@pytest.mark.asyncio
async def test_get_webhook_user_id_returns_installer_keycloak_id(
    webhook_store, sample_webhook
):
    user_id = await webhook_store.get_webhook_user_id(
        project_key='PROJ', repo_slug='myrepo'
    )
    assert user_id == 'kc-installer'
