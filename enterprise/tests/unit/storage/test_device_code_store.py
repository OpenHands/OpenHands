"""Unit tests for DeviceCodeStore."""

from unittest.mock import patch

import pytest
from sqlalchemy import select
from storage.device_code import DeviceCode
from storage.device_code_store import DeviceCodeStore


@pytest.fixture
def device_code_store():
    """Create DeviceCodeStore instance."""
    return DeviceCodeStore()


class TestDeviceCodeStore:
    """Test cases for DeviceCodeStore."""

    def test_generate_user_code(self, device_code_store):
        """Test user code generation."""
        code = device_code_store.generate_user_code()

        assert len(code) == 8
        assert code.isupper()
        # Should not contain confusing characters
        assert not any(char in code for char in 'IO01')

    def test_generate_device_code(self, device_code_store):
        """Test device code generation."""
        code = device_code_store.generate_device_code()

        assert len(code) == 128
        assert code.isalnum()

    @pytest.mark.asyncio
    async def test_create_device_code_success(
        self, device_code_store, async_session_maker
    ):
        """Test successful device code creation."""
        with patch('storage.device_code_store.a_session_maker', async_session_maker):
            result = await device_code_store.create_device_code(expires_in=600)

        assert isinstance(result, DeviceCode)
        assert len(result.device_code) == 128
        assert len(result.user_code) == 8

        # Verify the DeviceCode was created in the database
        async with async_session_maker() as session:
            result_db = await session.execute(
                select(DeviceCode).filter(DeviceCode.device_code == result.device_code)
            )
            device_code = result_db.scalars().first()
            assert device_code is not None
            assert device_code.user_code == result.user_code

    @pytest.mark.asyncio
    async def test_create_device_code_with_retries(
        self, device_code_store, async_session_maker
    ):
        """Test device code creation with constraint violation retries."""
        # First create a device code to cause a collision
        with patch('storage.device_code_store.a_session_maker', async_session_maker):
            first_code = await device_code_store.create_device_code(expires_in=600)

        # Patch generate methods to return the same codes on first attempt,
        # then different codes on second attempt
        call_count = {'user': 0, 'device': 0}
        original_generate_user_code = device_code_store.generate_user_code
        original_generate_device_code = device_code_store.generate_device_code

        def mock_generate_user_code():
            call_count['user'] += 1
            if call_count['user'] == 1:
                return first_code.user_code  # Collision
            return original_generate_user_code()

        def mock_generate_device_code():
            call_count['device'] += 1
            if call_count['device'] == 1:
                return first_code.device_code  # Collision
            return original_generate_device_code()

        device_code_store.generate_user_code = mock_generate_user_code
        device_code_store.generate_device_code = mock_generate_device_code

        with patch('storage.device_code_store.a_session_maker', async_session_maker):
            result = await device_code_store.create_device_code(expires_in=600)

        assert isinstance(result, DeviceCode)
        assert result.device_code != first_code.device_code  # Should be different
        assert call_count['user'] == 2  # Two attempts

    @pytest.mark.asyncio
    async def test_create_device_code_max_attempts_exceeded(
        self, device_code_store, async_session_maker
    ):
        """Test device code creation failure after max attempts."""
        # First create a device code
        with patch('storage.device_code_store.a_session_maker', async_session_maker):
            first_code = await device_code_store.create_device_code(expires_in=600)

        # Always return the same codes to cause repeated collisions
        device_code_store.generate_user_code = lambda: first_code.user_code
        device_code_store.generate_device_code = lambda: first_code.device_code

        with patch('storage.device_code_store.a_session_maker', async_session_maker):
            with pytest.raises(
                RuntimeError,
                match='Failed to generate unique device codes after 3 attempts',
            ):
                await device_code_store.create_device_code(
                    expires_in=600, max_attempts=3
                )

    @pytest.mark.asyncio
    async def test_get_by_device_code(self, device_code_store, async_session_maker):
        """Test getting device code by device code."""
        # Create a device code first
        with patch('storage.device_code_store.a_session_maker', async_session_maker):
            created = await device_code_store.create_device_code(expires_in=600)
            result = await device_code_store.get_by_device_code(created.device_code)

        assert result is not None
        assert result.device_code == created.device_code
        assert result.user_code == created.user_code

    @pytest.mark.asyncio
    async def test_get_by_device_code_not_found(
        self, device_code_store, async_session_maker
    ):
        """Test getting non-existent device code."""
        with patch('storage.device_code_store.a_session_maker', async_session_maker):
            result = await device_code_store.get_by_device_code('non-existent-code')

        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_user_code(self, device_code_store, async_session_maker):
        """Test getting device code by user code."""
        # Create a device code first
        with patch('storage.device_code_store.a_session_maker', async_session_maker):
            created = await device_code_store.create_device_code(expires_in=600)
            result = await device_code_store.get_by_user_code(created.user_code)

        assert result is not None
        assert result.device_code == created.device_code
        assert result.user_code == created.user_code

    @pytest.mark.asyncio
    async def test_get_by_user_code_not_found(
        self, device_code_store, async_session_maker
    ):
        """Test getting non-existent user code."""
        with patch('storage.device_code_store.a_session_maker', async_session_maker):
            result = await device_code_store.get_by_user_code('NOTFOUND')

        assert result is None

    @pytest.mark.asyncio
    async def test_authorize_device_code_success(
        self, device_code_store, async_session_maker
    ):
        """Test successful device code authorization."""
        user_id = 'test-user-123'

        with patch('storage.device_code_store.a_session_maker', async_session_maker):
            created = await device_code_store.create_device_code(expires_in=600)
            result = await device_code_store.authorize_device_code(
                created.user_code, user_id
            )

        assert result is True

        # Verify the device code was authorized in the database
        async with async_session_maker() as session:
            result_db = await session.execute(
                select(DeviceCode).filter(DeviceCode.user_code == created.user_code)
            )
            device_code = result_db.scalars().first()
            assert device_code.status == 'authorized'
            assert device_code.keycloak_user_id == user_id

    @pytest.mark.asyncio
    async def test_authorize_device_code_not_found(
        self, device_code_store, async_session_maker
    ):
        """Test authorizing non-existent device code."""
        with patch('storage.device_code_store.a_session_maker', async_session_maker):
            result = await device_code_store.authorize_device_code(
                'NOTFOUND', 'user-123'
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_authorize_device_code_not_pending(
        self, device_code_store, async_session_maker
    ):
        """Test authorizing already authorized device code."""
        user_id = 'test-user-123'

        with patch('storage.device_code_store.a_session_maker', async_session_maker):
            created = await device_code_store.create_device_code(expires_in=600)
            # First authorization
            await device_code_store.authorize_device_code(created.user_code, user_id)
            # Second authorization should fail
            result = await device_code_store.authorize_device_code(
                created.user_code, 'another-user'
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_deny_device_code_success(
        self, device_code_store, async_session_maker
    ):
        """Test successful device code denial."""
        with patch('storage.device_code_store.a_session_maker', async_session_maker):
            created = await device_code_store.create_device_code(expires_in=600)
            result = await device_code_store.deny_device_code(created.user_code)

        assert result is True

        # Verify the device code was denied in the database
        async with async_session_maker() as session:
            result_db = await session.execute(
                select(DeviceCode).filter(DeviceCode.user_code == created.user_code)
            )
            device_code = result_db.scalars().first()
            assert device_code.status == 'denied'

    @pytest.mark.asyncio
    async def test_deny_device_code_not_found(
        self, device_code_store, async_session_maker
    ):
        """Test denying non-existent device code."""
        with patch('storage.device_code_store.a_session_maker', async_session_maker):
            result = await device_code_store.deny_device_code('NOTFOUND')

        assert result is False

    @pytest.mark.asyncio
    async def test_deny_device_code_not_pending(
        self, device_code_store, async_session_maker
    ):
        """Test denying already denied device code."""
        with patch('storage.device_code_store.a_session_maker', async_session_maker):
            created = await device_code_store.create_device_code(expires_in=600)
            # First denial
            await device_code_store.deny_device_code(created.user_code)
            # Second denial should fail
            result = await device_code_store.deny_device_code(created.user_code)

        assert result is False

    @pytest.mark.asyncio
    async def test_update_poll_time_success(
        self, device_code_store, async_session_maker
    ):
        """Test updating poll time."""
        with patch('storage.device_code_store.a_session_maker', async_session_maker):
            created = await device_code_store.create_device_code(expires_in=600)
            original_interval = created.current_interval
            result = await device_code_store.update_poll_time(
                created.device_code, increase_interval=True
            )

        assert result is True

        # Verify the poll time was updated
        async with async_session_maker() as session:
            result_db = await session.execute(
                select(DeviceCode).filter(DeviceCode.device_code == created.device_code)
            )
            device_code = result_db.scalars().first()
            assert device_code.current_interval > original_interval

    @pytest.mark.asyncio
    async def test_update_poll_time_not_found(
        self, device_code_store, async_session_maker
    ):
        """Test updating poll time for non-existent device code."""
        with patch('storage.device_code_store.a_session_maker', async_session_maker):
            result = await device_code_store.update_poll_time(
                'non-existent-code', increase_interval=False
            )

        assert result is False
