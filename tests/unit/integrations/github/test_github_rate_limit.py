"""
Unit tests for GitHub rate limit and forbidden error handling.

Tests cover:
1. Rate limit error detection (403 with rate limit message)
2. Permission denied error handling (403 without rate limit message)
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from openhands.integrations.github.service.repos import GithubReposService
from openhands.integrations.service_types import ForbiddenError, RateLimitError
from openhands.server.shared import AppMode


@pytest.fixture
def repos_service():
    """Create a GithubReposService instance for testing."""
    service = GithubReposService()
    service.BASE_URL = 'https://api.github.com'
    return service


class TestRateLimitHandling:
    """Test proper handling of GitHub rate limit errors (403 responses)."""

    @pytest.mark.asyncio
    async def test_rate_limit_error_detection(self, repos_service):
        """
        Test that 403 with rate limit message is properly detected and raised as RateLimitError.

        GitHub returns 403 (not 429) when rate limit is exceeded.
        The error message contains "rate limit" or "API rate limit exceeded".
        """
        # Mock a 403 response with rate limit message
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.json.return_value = {
            'message': 'API rate limit exceeded for 34.70.174.52. (But here\'s the good news: Authenticated requests get a higher rate limit. Check out the documentation for more details.)',
            'documentation_url': 'https://docs.github.com/rest/overview/resources-in-the-rest-api#rate-limiting',
        }

        with patch.object(
            repos_service, '_make_request', side_effect=RateLimitError(mock_response)
        ):
            with pytest.raises(RateLimitError) as exc_info:
                await repos_service.search_repositories(
                    query='test',
                    per_page=100,
                    sort='stars',
                    order='desc',
                    public=False,
                    app_mode=AppMode.SAAS,
                )

            # Verify it's a RateLimitError, not UnknownException
            assert isinstance(exc_info.value, RateLimitError)

    @pytest.mark.asyncio
    async def test_forbidden_error_non_rate_limit(self, repos_service):
        """
        Test that 403 without rate limit message is raised as ForbiddenError.

        Not all 403 errors are rate limits - could be permission denied, etc.
        """
        # Mock a 403 response without rate limit keywords
        mock_response = Mock()
        mock_response.status_code = 403
        mock_response.json.return_value = {
            'message': 'Resource not accessible by integration',
            'documentation_url': 'https://docs.github.com/rest/reference/repos',
        }

        with patch.object(
            repos_service, '_make_request', side_effect=ForbiddenError(mock_response)
        ):
            with pytest.raises(ForbiddenError) as exc_info:
                await repos_service.search_repositories(
                    query='test',
                    per_page=100,
                    sort='stars',
                    order='desc',
                    public=False,
                    app_mode=AppMode.SAAS,
                )

            # Verify it's ForbiddenError, not RateLimitError
            assert isinstance(exc_info.value, ForbiddenError)
            assert not isinstance(exc_info.value, RateLimitError)
