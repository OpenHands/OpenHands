# mypy: disable-error-code="unreachable"
"""Tests for the GitLab resolver.
"""

import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.responses import JSONResponse
from server.routes.integration.gitlab import gitlab_events


@pytest.mark.asyncio
@patch('server.routes.integration.gitlab.verify_gitlab_signature')
@patch('server.routes.integration.gitlab.gitlab_manager')
@patch('server.routes.integration.gitlab.sio')
async def test_gitlab_events_deduplication_with_object_id(
    mock_sio, mock_gitlab_manager, mock_verify_signature
):
    """Test that duplicate GitLab events are deduplicated using object_attributes.id."""
    # Setup mocks
    mock_verify_signature.return_value = None
    mock_gitlab_manager.receive_message = AsyncMock()

    # Mock Redis
    mock_redis = AsyncMock()
    mock_sio.manager.redis = mock_redis

    # First request - Redis returns True (key was set)
    mock_redis.set.return_value = True

    # Create a mock request with a payload containing object_attributes.id
    payload = {
        'object_kind': 'note',
        'object_attributes': {
            'discussion_id': 'test_discussion_id',
            'note': '@openhands help me with this',
            'id': 12345,
        },
    }

    mock_request = MagicMock()
    mock_request.json = AsyncMock(return_value=payload)

    # Call the endpoint
    response = await gitlab_events(
        request=mock_request,
        x_gitlab_token='test_token',
        x_openhands_webhook_id='test_webhook_id',
        x_openhands_user_id='test_user_id',
    )

    # Verify Redis was called to set the key with the object_attributes.id
    mock_redis.set.assert_called_once_with(12345, 1, nx=True, ex=60)

    # Verify the message was processed
    assert mock_gitlab_manager.receive_message.called
    assert isinstance(response, JSONResponse)
    assert response.status_code == 200

    # Reset mocks
    mock_redis.set.reset_mock()
    mock_gitlab_manager.receive_message.reset_mock()

    # Second request - Redis returns False (key already exists)
    mock_redis.set.return_value = False

    # Call the endpoint again with the same payload
    response = await gitlab_events(
        request=mock_request,
        x_gitlab_token='test_token',
        x_openhands_webhook_id='test_webhook_id',
        x_openhands_user_id='test_user_id',
    )

    # Verify Redis was called to set the key with the object_attributes.id
    mock_redis.set.assert_called_once_with(12345, 1, nx=True, ex=60)

    # Verify the message was NOT processed (duplicate)
    assert not mock_gitlab_manager.receive_message.called
    assert isinstance(response, JSONResponse)
    assert response.status_code == 200
    # mypy: disable-error-code="unreachable"
    response_body = json.loads(response.body)  # type: ignore
    assert response_body['message'] == 'Duplicate GitLab event ignored.'


@pytest.mark.asyncio
@patch('server.routes.integration.gitlab.verify_gitlab_signature')
@patch('server.routes.integration.gitlab.gitlab_manager')
@patch('server.routes.integration.gitlab.sio')
async def test_gitlab_events_deduplication_without_object_id(
    mock_sio, mock_gitlab_manager, mock_verify_signature
):
    """Test that GitLab events without object_attributes.id are deduplicated using hash of payload."""
    # Setup mocks
    mock_verify_signature.return_value = None
    mock_gitlab_manager.receive_message = AsyncMock()

    # Mock Redis
    mock_redis = AsyncMock()
    mock_sio.manager.redis = mock_redis

    # First request - Redis returns True (key was set)
    mock_redis.set.return_value = True

    # Create a mock request with a payload without object_attributes.id
    payload = {
        'object_kind': 'pipeline',
        'object_attributes': {
            'ref': 'main',
            'status': 'success',
            # No 'id' field
        },
    }

    mock_request = MagicMock()
    mock_request.json = AsyncMock(return_value=payload)

    # Calculate the expected hash
    dedup_json = json.dumps(payload, sort_keys=True)
    expected_hash = hashlib.sha256(dedup_json.encode()).hexdigest()
    expected_key = f'gitlab_msg: {expected_hash}'  # Note the space after 'gitlab_msg:'

    # Call the endpoint
    response = await gitlab_events(
        request=mock_request,
        x_gitlab_token='test_token',
        x_openhands_webhook_id='test_webhook_id',
        x_openhands_user_id='test_user_id',
    )

    # Verify Redis was called to set the key with the hash
    mock_redis.set.assert_called_once_with(expected_key, 1, nx=True, ex=60)

    # Verify the message was processed
    assert mock_gitlab_manager.receive_message.called
    assert isinstance(response, JSONResponse)
    assert response.status_code == 200

    # Reset mocks
    mock_redis.set.reset_mock()
    mock_gitlab_manager.receive_message.reset_mock()

    # Second request - Redis returns False (key already exists)
    mock_redis.set.return_value = False

    # Call the endpoint again with the same payload
    response = await gitlab_events(
        request=mock_request,
        x_gitlab_token='test_token',
        x_openhands_webhook_id='test_webhook_id',
        x_openhands_user_id='test_user_id',
    )

    # Verify Redis was called to set the key with the hash
    mock_redis.set.assert_called_once_with(expected_key, 1, nx=True, ex=60)

    # Verify the message was NOT processed (duplicate)
    assert not mock_gitlab_manager.receive_message.called
    assert isinstance(response, JSONResponse)
    assert response.status_code == 200
    # mypy: disable-error-code="unreachable"
    response_body = json.loads(response.body)  # type: ignore
    assert response_body['message'] == 'Duplicate GitLab event ignored.'


@pytest.mark.asyncio
@patch('server.routes.integration.gitlab.verify_gitlab_signature')
@patch('server.routes.integration.gitlab.gitlab_manager')
@patch('server.routes.integration.gitlab.sio')
async def test_gitlab_events_different_payloads_not_deduplicated(
    mock_sio, mock_gitlab_manager, mock_verify_signature
):
    """Test that different GitLab events are not deduplicated."""
    # Setup mocks
    mock_verify_signature.return_value = None
    mock_gitlab_manager.receive_message = AsyncMock()

    # Mock Redis
    mock_redis = AsyncMock()
    mock_sio.manager.redis = mock_redis
    mock_redis.set.return_value = True  # Always return True for this test

    # First payload with ID 123
    payload1 = {
        'object_kind': 'issue',
        'object_attributes': {'id': 123, 'title': 'Test Issue', 'action': 'open'},
    }

    mock_request1 = MagicMock()
    mock_request1.json = AsyncMock(return_value=payload1)

    # Call the endpoint with first payload
    response1 = await gitlab_events(
        request=mock_request1,
        x_gitlab_token='test_token',
        x_openhands_webhook_id='test_webhook_id',
        x_openhands_user_id='test_user_id',
    )

    # Verify Redis was called to set the key with the first ID
    mock_redis.set.assert_called_once_with(123, 1, nx=True, ex=60)
    mock_redis.set.reset_mock()

    # Verify the first message was processed
    assert mock_gitlab_manager.receive_message.called
    assert isinstance(response1, JSONResponse)
    assert response1.status_code == 200
    mock_gitlab_manager.receive_message.reset_mock()

    # Second payload with different ID 456
    payload2 = {
        'object_kind': 'issue',
        'object_attributes': {'id': 456, 'title': 'Another Issue', 'action': 'open'},
    }

    mock_request2 = MagicMock()
    mock_request2.json = AsyncMock(return_value=payload2)

    # Call the endpoint with second payload
    response2 = await gitlab_events(
        request=mock_request2,
        x_gitlab_token='test_token',
        x_openhands_webhook_id='test_webhook_id',
        x_openhands_user_id='test_user_id',
    )

    # Verify Redis was called to set the key with the second ID
    mock_redis.set.assert_called_once_with(456, 1, nx=True, ex=60)

    # Verify the second message was also processed (not deduplicated)
    assert mock_gitlab_manager.receive_message.called
    assert isinstance(response2, JSONResponse)
    assert response2.status_code == 200


@pytest.mark.asyncio
@patch('server.routes.integration.gitlab.verify_gitlab_signature')
@patch('server.routes.integration.gitlab.gitlab_manager')
@patch('server.routes.integration.gitlab.sio')
async def test_gitlab_events_multiple_identical_payloads_deduplicated(
    mock_sio, mock_gitlab_manager, mock_verify_signature
):
    """Test that multiple identical GitLab events are properly deduplicated."""
    # Setup mocks
    mock_verify_signature.return_value = None
    mock_gitlab_manager.receive_message = AsyncMock()

    # Mock Redis
    mock_redis = AsyncMock()
    mock_sio.manager.redis = mock_redis

    # Create a payload with object_attributes.id
    payload = {
        'object_kind': 'merge_request',
        'object_attributes': {
            'id': 789,
            'title': 'Fix bug',
            'description': 'This fixes the bug',
            'state': 'opened',
        },
    }

    mock_request = MagicMock()
    mock_request.json = AsyncMock(return_value=payload)

    # First request - Redis returns True (key was set)
    mock_redis.set.return_value = True

    # Call the endpoint first time
    response1 = await gitlab_events(
        request=mock_request,
        x_gitlab_token='test_token',
        x_openhands_webhook_id='test_webhook_id',
        x_openhands_user_id='test_user_id',
    )

    # Verify Redis was called to set the key with the object_attributes.id
    mock_redis.set.assert_called_once_with(789, 1, nx=True, ex=60)
    mock_redis.set.reset_mock()

    # Verify the message was processed
    assert mock_gitlab_manager.receive_message.called
    assert isinstance(response1, JSONResponse)
    assert response1.status_code == 200
    assert (
        json.loads(response1.body)['message']
        == 'GitLab events endpoint reached successfully.'
    )
    mock_gitlab_manager.receive_message.reset_mock()

    # Second request - Redis returns False (key already exists)
    mock_redis.set.return_value = False

    # Call the endpoint second time with the same payload
    response2 = await gitlab_events(
        request=mock_request,
        x_gitlab_token='test_token',
        x_openhands_webhook_id='test_webhook_id',
        x_openhands_user_id='test_user_id',
    )

    # Verify Redis was called to set the key with the same object_attributes.id
    mock_redis.set.assert_called_once_with(789, 1, nx=True, ex=60)
    mock_redis.set.reset_mock()

    # Verify the message was NOT processed (duplicate)
    assert not mock_gitlab_manager.receive_message.called
    assert isinstance(response2, JSONResponse)
    assert response2.status_code == 200
    # mypy: disable-error-code="unreachable"
    response2_body = json.loads(response2.body)  # type: ignore
    assert response2_body['message'] == 'Duplicate GitLab event ignored.'

    # Third request - Redis returns False again (key still exists)
    mock_redis.set.return_value = False

    # Call the endpoint third time with the same payload
    response3 = await gitlab_events(
        request=mock_request,
        x_gitlab_token='test_token',
        x_openhands_webhook_id='test_webhook_id',
        x_openhands_user_id='test_user_id',
    )

    # Verify Redis was called to set the key with the same object_attributes.id
    mock_redis.set.assert_called_once_with(789, 1, nx=True, ex=60)

    # Verify the message was NOT processed (duplicate)
    assert not mock_gitlab_manager.receive_message.called
    assert isinstance(response3, JSONResponse)
    assert response3.status_code == 200
    # mypy: disable-error-code="unreachable"
    response3_body = json.loads(response3.body)  # type: ignore
    assert response3_body['message'] == 'Duplicate GitLab event ignored.'
