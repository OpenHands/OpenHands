from unittest.mock import patch

import pytest
from litellm.exceptions import APIConnectionError, BadGatewayError

from openhands.core.config import LLMConfig
from openhands.llm.llm import LLM


@pytest.fixture
def default_config():
    return LLMConfig(
        model='gpt-4o',
        api_key='test_key',
        num_retries=2,
        retry_min_wait=1,
        retry_max_wait=2,
    )


@patch('openhands.llm.llm.litellm_completion')
def test_completion_retries_api_connection_error(
    mock_litellm_completion, default_config
):
    """Test that APIConnectionError is properly retried."""
    # Mock the litellm_completion to first raise an APIConnectionError, then return a successful response
    mock_litellm_completion.side_effect = [
        APIConnectionError(
            message='API connection error',
            llm_provider='test_provider',
            model='test_model',
        ),
        {'choices': [{'message': {'content': 'Retry successful'}}]},
    ]

    # Create an LLM instance and call completion
    llm = LLM(config=default_config, service_id='test-service')
    response = llm.completion(
        messages=[{'role': 'user', 'content': 'Hello!'}],
        stream=False,
    )

    # Verify that the retry was successful
    assert response['choices'][0]['message']['content'] == 'Retry successful'
    assert mock_litellm_completion.call_count == 2  # Initial call + 1 retry


@patch('openhands.llm.llm.litellm_completion')
def test_completion_max_retries_api_connection_error(
    mock_litellm_completion, default_config
):
    """Test that APIConnectionError respects max retries."""
    # Mock the litellm_completion to raise APIConnectionError multiple times
    mock_litellm_completion.side_effect = [
        APIConnectionError(
            message='API connection error 1',
            llm_provider='test_provider',
            model='test_model',
        ),
        APIConnectionError(
            message='API connection error 2',
            llm_provider='test_provider',
            model='test_model',
        ),
        APIConnectionError(
            message='API connection error 3',
            llm_provider='test_provider',
            model='test_model',
        ),
    ]

    # Create an LLM instance and call completion
    llm = LLM(config=default_config, service_id='test-service')

    # The completion should raise an APIConnectionError after exhausting all retries
    with pytest.raises(APIConnectionError) as excinfo:
        llm.completion(
            messages=[{'role': 'user', 'content': 'Hello!'}],
            stream=False,
        )

    # Verify that the correct number of retries were attempted
    # The actual behavior is that it tries the initial call + num_retries (not +1)
    assert mock_litellm_completion.call_count == default_config.num_retries

    # The exception doesn't contain retry information in the current implementation
    # Just verify that we got an APIConnectionError
    assert 'API connection error' in str(excinfo.value)


@patch('openhands.llm.llm.litellm_completion')
def test_completion_retries_bad_gateway_error(mock_litellm_completion, default_config):
    """Test that BadGatewayError is properly retried."""
    mock_litellm_completion.side_effect = [
        BadGatewayError(
            message='Bad gateway',
            llm_provider='test_provider',
            model='test_model',
        ),
        {'choices': [{'message': {'content': 'Retry successful'}}]},
    ]

    llm = LLM(config=default_config, service_id='test-service')
    response = llm.completion(
        messages=[{'role': 'user', 'content': 'Hello!'}],
        stream=False,
    )

    assert response['choices'][0]['message']['content'] == 'Retry successful'
    assert mock_litellm_completion.call_count == 2
