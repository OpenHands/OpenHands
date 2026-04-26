"""Tests for BitBucketPRsMixin: create_pr URL extraction with null nested fields."""

from unittest.mock import patch

import pytest
from pydantic import SecretStr

from openhands.app_server.integrations.bitbucket.bitbucket_service import (
    BitBucketService,
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'response_links, expected_url',
    [
        pytest.param(
            {'html': {'href': 'https://bitbucket.org/w/r/pull-requests/1'}},
            'https://bitbucket.org/w/r/pull-requests/1',
            id='happy_path',
        ),
        pytest.param(None, '', id='links_null'),
        pytest.param({}, '', id='links_empty_dict'),
        pytest.param({'html': None}, '', id='html_null'),
        pytest.param({'html': {}}, '', id='html_empty_dict'),
        pytest.param('not-a-dict', '', id='links_non_dict'),
        pytest.param({'html': 'not-a-dict'}, '', id='html_non_dict'),
    ],
)
async def test_create_pr_handles_null_links_or_html(response_links, expected_url):
    """Bitbucket cloud's create-PR response can carry ``"links": null`` or
    ``"html": null`` when responses pass through corporate proxies that
    strip empty objects. The original
    ``data.get('links', {}).get('html', {}).get('href', '')`` chain crashed
    with ``AttributeError`` on explicit nulls — meaning a successful PR
    creation surfaced an exception to the user instead of returning the
    new PR's URL. The fix mirrors the defensive pattern landed for
    ``get_user`` in #14070 and ``_parse_repository`` in #14085."""
    service = BitBucketService(token=SecretStr('t'))
    response = {'links': response_links}

    with patch.object(service, '_make_request', return_value=(response, {})):
        url = await service.create_pr(
            repo_name='w/r',
            source_branch='feat',
            target_branch='main',
            title='t',
            body='b',
            draft=False,
        )

    assert url == expected_url
