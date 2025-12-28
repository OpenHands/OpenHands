"""Unit tests for TelemetryMetrics model."""

import uuid
from datetime import UTC, datetime

from storage.telemetry_metrics import TelemetryMetrics


class TestTelemetryMetrics:
    """Test cases for TelemetryMetrics model."""

    def test_init_with_metrics_data(self):
        """Test initialization with metrics data."""
        metrics_data = {
            'cpu_usage': 75.5,
            'memory_usage': 1024,
            'active_sessions': 5,
        }

        metrics = TelemetryMetrics(metrics_data=metrics_data)

        assert metrics.metrics_data == metrics_data
        assert metrics.upload_attempts == 0
        assert metrics.uploaded_at is None
        assert metrics.last_upload_error is None
        assert metrics.collected_at is not None
        assert metrics.created_at is not None
        assert metrics.updated_at is not None

    def test_init_with_custom_collected_at(self):
        """Test initialization with custom collected_at timestamp."""
        metrics_data = {'test': 'value'}
        custom_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC)

        metrics = TelemetryMetrics(metrics_data=metrics_data, collected_at=custom_time)

        assert metrics.collected_at == custom_time

    def test_mark_uploaded(self):
        """Test marking metrics as uploaded."""
        metrics = TelemetryMetrics(metrics_data={'test': 'data'})

        # Initially not uploaded
        assert not metrics.is_uploaded
        assert metrics.uploaded_at is None

        # Mark as uploaded
        metrics.mark_uploaded()

        assert metrics.is_uploaded

    def test_mark_upload_failed(self):
        """Test marking upload as failed."""
        metrics = TelemetryMetrics(metrics_data={'test': 'data'})
        error_message = 'Network timeout'

        # Initially no failures
        assert metrics.upload_attempts == 0
        assert metrics.last_upload_error is None

        # Mark as failed
        metrics.mark_upload_failed(error_message)

        assert metrics.upload_attempts == 1
        assert metrics.last_upload_error == error_message
        assert metrics.uploaded_at is None
        assert not metrics.is_uploaded

    def test_multiple_upload_failures(self):
        """Test multiple upload failures increment attempts."""
        metrics = TelemetryMetrics(metrics_data={'test': 'data'})

        metrics.mark_upload_failed('Error 1')
        assert metrics.upload_attempts == 1

        metrics.mark_upload_failed('Error 2')
        assert metrics.upload_attempts == 2
        assert metrics.last_upload_error == 'Error 2'

    def test_is_uploaded_property(self):
        """Test is_uploaded property."""
        metrics = TelemetryMetrics(metrics_data={'test': 'data'})

        # Initially not uploaded
        assert not metrics.is_uploaded

        # After marking uploaded
        metrics.mark_uploaded()
        assert metrics.is_uploaded

    def test_needs_retry_property(self):
        """Test needs_retry property logic."""
        metrics = TelemetryMetrics(metrics_data={'test': 'data'})

        # Initially needs retry (0 attempts, not uploaded)
        assert metrics.needs_retry

        # After 1 failure, still needs retry
        metrics.mark_upload_failed('Error 1')
        assert metrics.needs_retry

        # After 2 failures, still needs retry
        metrics.mark_upload_failed('Error 2')
        assert metrics.needs_retry

        # After 3 failures, no more retries
        metrics.mark_upload_failed('Error 3')
        assert not metrics.needs_retry

        # Reset and test successful upload
        metrics2 = TelemetryMetrics(metrics_data={'test': 'data'})  # type: ignore[unreachable]
        metrics2.mark_uploaded()
        # After upload, needs_retry should be False since is_uploaded is True

    def test_upload_failure_clears_uploaded_at(self):
        """Test that upload failure clears uploaded_at timestamp."""
        metrics = TelemetryMetrics(metrics_data={'test': 'data'})

        # Mark as uploaded first
        metrics.mark_uploaded()
        assert metrics.uploaded_at is not None

        # Mark as failed - should clear uploaded_at
        metrics.mark_upload_failed('Network error')
        assert metrics.uploaded_at is None

    def test_successful_upload_clears_error(self):
        """Test that successful upload clears error message."""
        metrics = TelemetryMetrics(metrics_data={'test': 'data'})

        # Mark as failed first
        metrics.mark_upload_failed('Network error')
        assert metrics.last_upload_error == 'Network error'

        # Mark as uploaded - should clear error
        metrics.mark_uploaded()
        assert metrics.last_upload_error is None

    def test_uuid_generation(self):
        """Test that each instance gets a unique UUID."""
        metrics1 = TelemetryMetrics(metrics_data={'test': 'data1'})
        metrics2 = TelemetryMetrics(metrics_data={'test': 'data2'})

        assert metrics1.id != metrics2.id
        assert isinstance(uuid.UUID(metrics1.id), uuid.UUID)
        assert isinstance(uuid.UUID(metrics2.id), uuid.UUID)

    def test_repr(self):
        """Test string representation."""
        metrics = TelemetryMetrics(metrics_data={'test': 'data'})
        repr_str = repr(metrics)

        assert 'TelemetryMetrics' in repr_str
        assert metrics.id in repr_str
        assert str(metrics.collected_at) in repr_str
        assert 'uploaded=False' in repr_str

        # Test after upload
        metrics.mark_uploaded()
        repr_str = repr(metrics)
        assert 'uploaded=True' in repr_str

    def test_complex_metrics_data(self):
        """Test with complex nested metrics data."""
        complex_data = {
            'system': {
                'cpu': {'usage': 75.5, 'cores': 8},
                'memory': {'total': 16384, 'used': 8192},
            },
            'sessions': [
                {'id': 'session1', 'duration': 3600},
                {'id': 'session2', 'duration': 1800},
            ],
            'timestamp': '2023-01-01T12:00:00Z',
        }

        metrics = TelemetryMetrics(metrics_data=complex_data)

        assert metrics.metrics_data == complex_data

    def test_empty_metrics_data(self):
        """Test with empty metrics data."""
        metrics = TelemetryMetrics(metrics_data={})

        assert metrics.metrics_data == {}

    def test_config_class(self):
        """Test that Config class is properly set."""
        assert hasattr(TelemetryMetrics, 'Config')
        assert TelemetryMetrics.Config.from_attributes is True
