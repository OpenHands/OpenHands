"""Tests for openhands.llm.bedrock — Bedrock model discovery with default credential chain."""

from unittest.mock import MagicMock, patch

import pytest

from openhands.llm.bedrock import (
    _create_bedrock_client,
    list_foundation_models,
    remove_error_modelId,
)


# ---------------------------------------------------------------------------
# _create_bedrock_client
# ---------------------------------------------------------------------------

class TestCreateBedrockClient:
    """Verify the helper passes explicit creds when provided, otherwise relies
    on the default credential chain."""

    @patch('openhands.llm.bedrock.boto3.client')
    def test_explicit_credentials(self, mock_boto_client):
        _create_bedrock_client(
            aws_region_name='us-west-2',
            aws_access_key_id='AKID',
            aws_secret_access_key='SECRET',
        )
        mock_boto_client.assert_called_once_with(
            service_name='bedrock',
            region_name='us-west-2',
            aws_access_key_id='AKID',
            aws_secret_access_key='SECRET',
        )

    @patch('openhands.llm.bedrock.boto3.client')
    def test_default_chain_no_creds(self, mock_boto_client):
        """When no explicit creds are given, only service_name should be passed
        so boto3 falls back to its default credential chain."""
        _create_bedrock_client()
        mock_boto_client.assert_called_once_with(service_name='bedrock')

    @patch('openhands.llm.bedrock.boto3.client')
    def test_region_only(self, mock_boto_client):
        _create_bedrock_client(aws_region_name='eu-west-1')
        mock_boto_client.assert_called_once_with(
            service_name='bedrock', region_name='eu-west-1'
        )

    @patch('openhands.llm.bedrock.boto3.client')
    def test_partial_creds_ignored(self, mock_boto_client):
        """If only one of key/secret is set, neither should be passed."""
        _create_bedrock_client(aws_access_key_id='AKID')
        mock_boto_client.assert_called_once_with(service_name='bedrock')


# ---------------------------------------------------------------------------
# list_foundation_models
# ---------------------------------------------------------------------------

def _mock_client_with_models(model_ids, inference_profiles=None):
    """Return a mock boto3 Bedrock client that returns the given models."""
    client = MagicMock()
    client.list_foundation_models.return_value = {
        'modelSummaries': [{'modelId': mid} for mid in model_ids],
    }

    if inference_profiles is not None:
        paginator = MagicMock()
        paginator.paginate.return_value = [
            {'inferenceProfileSummaries': inference_profiles}
        ]
        client.get_paginator.return_value = paginator
    else:
        # Simulate missing ListInferenceProfiles permission
        paginator = MagicMock()
        paginator.paginate.side_effect = Exception('AccessDeniedException')
        client.get_paginator.return_value = paginator

    return client


class TestListFoundationModels:
    @patch('openhands.llm.bedrock._create_bedrock_client')
    def test_returns_foundation_models(self, mock_create):
        mock_create.return_value = _mock_client_with_models(
            ['anthropic.claude-v2', 'amazon.titan-text-express-v1']
        )
        result = list_foundation_models(aws_region_name='us-east-1')
        assert 'bedrock/anthropic.claude-v2' in result
        assert 'bedrock/amazon.titan-text-express-v1' in result

    @patch('openhands.llm.bedrock._create_bedrock_client')
    def test_includes_inference_profiles(self, mock_create):
        mock_create.return_value = _mock_client_with_models(
            ['anthropic.claude-v2'],
            inference_profiles=[
                {'inferenceProfileId': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0'},
            ],
        )
        result = list_foundation_models()
        assert 'bedrock/anthropic.claude-v2' in result
        assert (
            'bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0' in result
        )

    @patch('openhands.llm.bedrock._create_bedrock_client')
    def test_deduplicates_models(self, mock_create):
        """Foundation model and inference profile with same ID → one entry."""
        mock_create.return_value = _mock_client_with_models(
            ['anthropic.claude-v2'],
            inference_profiles=[
                {'inferenceProfileId': 'anthropic.claude-v2'},
            ],
        )
        result = list_foundation_models()
        assert result.count('bedrock/anthropic.claude-v2') == 1

    @patch('openhands.llm.bedrock._create_bedrock_client')
    def test_inference_profiles_failure_nonfatal(self, mock_create):
        """If ListInferenceProfiles fails, foundation models are still returned."""
        mock_create.return_value = _mock_client_with_models(
            ['anthropic.claude-v2'],
            inference_profiles=None,  # triggers AccessDeniedException
        )
        result = list_foundation_models()
        assert result == ['bedrock/anthropic.claude-v2']

    @patch('openhands.llm.bedrock._create_bedrock_client')
    def test_api_failure_returns_empty(self, mock_create):
        mock_create.side_effect = Exception('NoCredentialsError')
        result = list_foundation_models()
        assert result == []

    @patch('openhands.llm.bedrock._create_bedrock_client')
    def test_default_chain_called_without_creds(self, mock_create):
        mock_create.return_value = _mock_client_with_models(['m1'])
        list_foundation_models()
        mock_create.assert_called_once_with(None, None, None)

    @patch('openhands.llm.bedrock._create_bedrock_client')
    def test_explicit_creds_forwarded(self, mock_create):
        mock_create.return_value = _mock_client_with_models(['m1'])
        list_foundation_models('us-west-2', 'AK', 'SK')
        mock_create.assert_called_once_with('us-west-2', 'AK', 'SK')


# ---------------------------------------------------------------------------
# remove_error_modelId
# ---------------------------------------------------------------------------

class TestRemoveErrorModelId:
    def test_removes_bedrock_prefixed(self):
        assert remove_error_modelId(['bedrock/foo', 'gpt-4', 'bedrock/bar']) == [
            'gpt-4'
        ]

    def test_keeps_non_bedrock(self):
        assert remove_error_modelId(['gpt-4', 'claude-3']) == ['gpt-4', 'claude-3']
