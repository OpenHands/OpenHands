"""Tests for BitbucketDCPRsMixin: create_pr, get_pr_details, is_pr_open."""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import SecretStr

from openhands.integrations.bitbucket_data_center.bitbucket_dc_service import (
    BitbucketDCService,
)


def make_service():
    return BitbucketDCService(token=SecretStr('tok'), base_domain='host.example.com')


# ── create_pr ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_pr_payload_structure():
    svc = make_service()
    mock_response = {
        'id': 1,
        'links': {'self': [{'href': 'https://host.example.com/pr/1'}]},
    }

    with patch.object(
        svc, '_make_request', return_value=(mock_response, {})
    ) as mock_req:
        await svc.create_pr('PROJ/myrepo', 'feature', 'main', 'My PR')

    # The payload is passed as the 'params' positional arg
    payload = mock_req.call_args[1].get('params') or mock_req.call_args[0][1]
    assert payload['fromRef']['id'] == 'refs/heads/feature'
    assert payload['toRef']['id'] == 'refs/heads/main'
    assert payload['fromRef']['repository']['slug'] == 'myrepo'
    assert payload['fromRef']['repository']['project']['key'] == 'PROJ'


@pytest.mark.asyncio
async def test_create_pr_returns_href():
    svc = make_service()
    mock_response = {
        'id': 5,
        'links': {'self': [{'href': 'https://host.example.com/pr/5'}]},
    }

    with patch.object(svc, '_make_request', return_value=(mock_response, {})):
        url = await svc.create_pr('PROJ/myrepo', 'feature', 'main', 'My PR')

    assert url == 'https://host.example.com/pr/5'


@pytest.mark.asyncio
async def test_create_pr_html_link_dict():
    svc = make_service()
    mock_response = {
        'id': 5,
        'links': {'html': {'href': 'https://host.example.com/pr/5/html'}},
    }

    with patch.object(svc, '_make_request', return_value=(mock_response, {})):
        url = await svc.create_pr('PROJ/myrepo', 'feature', 'main', 'My PR')

    assert url == 'https://host.example.com/pr/5/html'


@pytest.mark.asyncio
async def test_create_pr_no_link_returns_empty_string():
    svc = make_service()
    mock_response = {'id': 5, 'links': {}}

    with patch.object(svc, '_make_request', return_value=(mock_response, {})):
        url = await svc.create_pr('PROJ/myrepo', 'feature', 'main', 'My PR')

    assert url == ''


# ── get_pr_details ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_pr_details_returns_raw_data():
    svc = make_service()
    mock_data = {'id': 3, 'state': 'OPEN', 'title': 'A PR'}

    with patch.object(svc, '_make_request', return_value=(mock_data, {})):
        result = await svc.get_pr_details('PROJ/myrepo', 3)

    assert result == mock_data


# ── is_pr_open ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_is_pr_open_returns_true():
    svc = make_service()

    with patch.object(
        svc, 'get_pr_details', new=AsyncMock(return_value={'state': 'OPEN'})
    ):
        assert await svc.is_pr_open('PROJ/myrepo', 1) is True


@pytest.mark.asyncio
async def test_is_pr_open_returns_false_for_merged():
    svc = make_service()

    with patch.object(
        svc, 'get_pr_details', new=AsyncMock(return_value={'state': 'MERGED'})
    ):
        assert await svc.is_pr_open('PROJ/myrepo', 1) is False


@pytest.mark.asyncio
async def test_is_pr_open_returns_false_for_declined():
    svc = make_service()

    with patch.object(
        svc, 'get_pr_details', new=AsyncMock(return_value={'state': 'DECLINED'})
    ):
        assert await svc.is_pr_open('PROJ/myrepo', 1) is False


@pytest.mark.asyncio
async def test_is_pr_open_returns_true_on_exception():
    """Current implementation catches all exceptions and returns True."""
    svc = make_service()

    with patch.object(
        svc,
        'get_pr_details',
        new=AsyncMock(side_effect=Exception('Some error')),
    ):
        result = await svc.is_pr_open('PROJ/myrepo', 999)
    assert result is True
