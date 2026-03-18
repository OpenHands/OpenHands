"""Tests for shared_event_router provider selection.

This module tests the get_shared_event_service_injector function which
determines which SharedEventServiceInjector to use based on environment variables.
"""

import os
from unittest.mock import patch

from server.sharing.aws_shared_event_service import AwsSharedEventServiceInjector
from server.sharing.google_cloud_shared_event_service import (
    GoogleCloudSharedEventServiceInjector,
)
from server.sharing.shared_event_router import get_shared_event_service_injector


class TestGetSharedEventServiceInjector:
    """Test cases for get_shared_event_service_injector function."""

    def test_defaults_to_google_cloud_when_no_env_set(self):
        """Test that GoogleCloudSharedEventServiceInjector is used when no env is set."""
        with patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            os.environ.pop('SHARED_EVENT_STORAGE_PROVIDER', None)
            os.environ.pop('FILE_STORE', None)

            injector = get_shared_event_service_injector()

            assert isinstance(injector, GoogleCloudSharedEventServiceInjector)

    def test_uses_google_cloud_when_file_store_google_cloud(self):
        """Test that GoogleCloudSharedEventServiceInjector is used when FILE_STORE=google_cloud."""
        with patch.dict(
            os.environ,
            {
                'FILE_STORE': 'google_cloud',
            },
            clear=True,
        ):
            os.environ.pop('SHARED_EVENT_STORAGE_PROVIDER', None)

            injector = get_shared_event_service_injector()

            assert isinstance(injector, GoogleCloudSharedEventServiceInjector)

    def test_uses_aws_when_provider_aws(self):
        """Test that AwsSharedEventServiceInjector is used when SHARED_EVENT_STORAGE_PROVIDER=aws."""
        with patch.dict(
            os.environ,
            {
                'SHARED_EVENT_STORAGE_PROVIDER': 'aws',
            },
            clear=True,
        ):
            injector = get_shared_event_service_injector()

            assert isinstance(injector, AwsSharedEventServiceInjector)

    def test_uses_gcp_when_provider_gcp(self):
        """Test that GoogleCloudSharedEventServiceInjector is used when SHARED_EVENT_STORAGE_PROVIDER=gcp."""
        with patch.dict(
            os.environ,
            {
                'SHARED_EVENT_STORAGE_PROVIDER': 'gcp',
            },
            clear=True,
        ):
            injector = get_shared_event_service_injector()

            assert isinstance(injector, GoogleCloudSharedEventServiceInjector)

    def test_uses_gcp_when_provider_google_cloud(self):
        """Test that GoogleCloudSharedEventServiceInjector is used when SHARED_EVENT_STORAGE_PROVIDER=google_cloud."""
        with patch.dict(
            os.environ,
            {
                'SHARED_EVENT_STORAGE_PROVIDER': 'google_cloud',
            },
            clear=True,
        ):
            injector = get_shared_event_service_injector()

            assert isinstance(injector, GoogleCloudSharedEventServiceInjector)

    def test_provider_takes_precedence_over_file_store(self):
        """Test that SHARED_EVENT_STORAGE_PROVIDER takes precedence over FILE_STORE."""
        with patch.dict(
            os.environ,
            {
                'SHARED_EVENT_STORAGE_PROVIDER': 'aws',
                'FILE_STORE': 'google_cloud',
            },
            clear=True,
        ):
            injector = get_shared_event_service_injector()

            # Should use AWS because SHARED_EVENT_STORAGE_PROVIDER takes precedence
            assert isinstance(injector, AwsSharedEventServiceInjector)

    def test_provider_gcp_takes_precedence_over_file_store_s3(self):
        """Test that SHARED_EVENT_STORAGE_PROVIDER=gcp takes precedence over FILE_STORE=s3."""
        with patch.dict(
            os.environ,
            {
                'SHARED_EVENT_STORAGE_PROVIDER': 'gcp',
                'FILE_STORE': 's3',
            },
            clear=True,
        ):
            injector = get_shared_event_service_injector()

            # Should use GCP because SHARED_EVENT_STORAGE_PROVIDER takes precedence
            assert isinstance(injector, GoogleCloudSharedEventServiceInjector)

    def test_provider_is_case_insensitive_aws(self):
        """Test that SHARED_EVENT_STORAGE_PROVIDER is case insensitive for AWS."""
        with patch.dict(
            os.environ,
            {
                'SHARED_EVENT_STORAGE_PROVIDER': 'AWS',
            },
            clear=True,
        ):
            injector = get_shared_event_service_injector()

            assert isinstance(injector, AwsSharedEventServiceInjector)

    def test_provider_is_case_insensitive_gcp(self):
        """Test that SHARED_EVENT_STORAGE_PROVIDER is case insensitive for GCP."""
        with patch.dict(
            os.environ,
            {
                'SHARED_EVENT_STORAGE_PROVIDER': 'GCP',
            },
            clear=True,
        ):
            injector = get_shared_event_service_injector()

            assert isinstance(injector, GoogleCloudSharedEventServiceInjector)

    def test_unknown_provider_defaults_to_google_cloud(self):
        """Test that unknown provider defaults to GoogleCloudSharedEventServiceInjector."""
        with patch.dict(
            os.environ,
            {
                'SHARED_EVENT_STORAGE_PROVIDER': 'unknown_provider',
            },
            clear=True,
        ):
            injector = get_shared_event_service_injector()

            # Should default to GCP for unknown providers
            assert isinstance(injector, GoogleCloudSharedEventServiceInjector)

    def test_empty_provider_falls_back_to_file_store(self):
        """Test that empty SHARED_EVENT_STORAGE_PROVIDER falls back to FILE_STORE."""
        with patch.dict(
            os.environ,
            {
                'SHARED_EVENT_STORAGE_PROVIDER': '',
                'FILE_STORE': 'google_cloud',
            },
            clear=True,
        ):
            injector = get_shared_event_service_injector()

            # Should default to GCP for unknown providers
            assert isinstance(injector, GoogleCloudSharedEventServiceInjector)
