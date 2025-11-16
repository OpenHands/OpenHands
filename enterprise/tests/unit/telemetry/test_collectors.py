"""Tests for the example collectors."""

from unittest.mock import MagicMock, patch

import pytest

from enterprise.telemetry.collectors.health_check import HealthCheckCollector
from enterprise.telemetry.collectors.system_metrics import SystemMetricsCollector
from enterprise.telemetry.collectors.user_activity import UserActivityCollector


class TestSystemMetricsCollector:
    """Test cases for the SystemMetricsCollector."""

    def setup_method(self):
        """Set up for each test."""
        self.collector = SystemMetricsCollector()

    def test_collector_name(self):
        """Test that collector has the correct name."""
        assert self.collector.collector_name == 'system_metrics'

    @patch('enterprise.telemetry.collectors.system_metrics.session_maker')
    def test_collect_success(self, mock_session_maker):
        """Test successful metrics collection."""
        # Mock database session and queries
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session

        # Mock different queries with different return values
        count_values = [
            100,
            50,
            1000,
            150,
            75,
            45,
        ]  # Different values for different queries
        count_call_index = 0

        def mock_count():
            nonlocal count_call_index
            value = count_values[count_call_index % len(count_values)]
            count_call_index += 1
            return value

        # Set up the mock chain
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.distinct.return_value = mock_query
        mock_query.count.side_effect = mock_count

        results = self.collector.collect()

        # Verify we got the expected metrics
        assert len(results) >= 6

        result_dict = {r.key: r.value for r in results}
        assert result_dict['total_users'] == 100
        assert result_dict['active_users'] == 50
        assert result_dict['total_conversations'] == 1000
        assert result_dict['conversations_30d'] == 150
        assert result_dict['conversations_7d'] == 75
        assert result_dict['active_users_30d'] == 45

    @patch('enterprise.telemetry.collectors.system_metrics.session_maker')
    def test_collect_database_error(self, mock_session_maker):
        """Test collection when database query fails."""
        mock_session_maker.return_value.__enter__.side_effect = Exception('DB Error')

        with pytest.raises(Exception, match='DB Error'):
            self.collector.collect()

    @patch('enterprise.telemetry.collectors.system_metrics.logger')
    @patch('enterprise.telemetry.collectors.system_metrics.session_maker')
    def test_collect_logs_success(self, mock_session_maker, mock_logger):
        """Test that successful collection is logged."""
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.count.return_value = 10

        # Mock the filter chain
        mock_query_chain = MagicMock()
        mock_query_chain.count.return_value = 5
        mock_session.query.return_value.filter.return_value = mock_query_chain
        mock_session.query.return_value.distinct.return_value = mock_query_chain

        self.collector.collect()

        mock_logger.info.assert_called()
        log_call = mock_logger.info.call_args[0][0]
        assert 'Collected' in log_call
        assert 'system metrics' in log_call


class TestUserActivityCollector:
    """Test cases for the UserActivityCollector."""

    def setup_method(self):
        """Set up for each test."""
        self.collector = UserActivityCollector()

    def test_collector_name(self):
        """Test that collector has the correct name."""
        assert self.collector.collector_name == 'user_activity'

    @patch('enterprise.telemetry.collectors.user_activity.session_maker')
    def test_collect_success(self, mock_session_maker):
        """Test successful user activity metrics collection."""
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session

        # Mock the query chain to return specific values for different queries
        # We'll use a counter to return different values for different calls
        count_values = [
            10,
            50,
            8,
        ]  # active_users_30d, conversations_30d, analytics_consent
        count_call_index = 0

        def mock_count():
            nonlocal count_call_index
            value = count_values[count_call_index % len(count_values)]
            count_call_index += 1
            return value

        # Set up the mock chain
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.distinct.return_value = mock_query
        mock_query.count.side_effect = mock_count

        # Mock for model usage query
        mock_query.group_by.return_value.order_by.return_value.limit.return_value.all.return_value = [
            ('gpt-4', 25),
            ('claude-3', 15),
        ]

        # Mock for provider usage query
        mock_query.group_by.return_value.all.return_value = [
            ('github', 30),
            ('gitlab', 10),
        ]

        # Mock for token stats query
        token_stats = MagicMock()
        token_stats.avg_tokens = 1500.0
        token_stats.total_tokens = 75000.0
        mock_query.first.return_value = token_stats

        results = self.collector.collect()

        # Verify we got metrics
        assert len(results) > 0

        result_dict = {r.key: r.value for r in results}
        assert 'avg_conversations_per_user_30d' in result_dict
        assert result_dict['avg_conversations_per_user_30d'] == 5.0  # 50/10

    @patch('enterprise.telemetry.collectors.user_activity.session_maker')
    def test_collect_with_zero_active_users(self, mock_session_maker):
        """Test collection when there are no active users."""
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session

        # Set up the mock chain
        mock_query = MagicMock()
        mock_session.query.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.distinct.return_value = mock_query
        mock_query.count.return_value = 0

        # Mock empty results for other queries
        mock_query.group_by.return_value.order_by.return_value.limit.return_value.all.return_value = []
        mock_query.group_by.return_value.all.return_value = []
        mock_query.first.return_value = None

        results = self.collector.collect()

        result_dict = {r.key: r.value for r in results}
        assert result_dict['avg_conversations_per_user_30d'] == 0

    @patch('enterprise.telemetry.collectors.user_activity.session_maker')
    def test_collect_database_error(self, mock_session_maker):
        """Test collection when database query fails."""
        mock_session_maker.return_value.__enter__.side_effect = Exception('DB Error')

        with pytest.raises(Exception, match='DB Error'):
            self.collector.collect()


class TestHealthCheckCollector:
    """Test cases for the HealthCheckCollector."""

    def setup_method(self):
        """Set up for each test."""
        self.collector = HealthCheckCollector()

    def test_collector_name(self):
        """Test that collector has the correct name."""
        assert self.collector.collector_name == 'health_check'

    @patch('enterprise.telemetry.collectors.health_check.session_maker')
    @patch('enterprise.telemetry.collectors.health_check.os.getenv')
    @patch('enterprise.telemetry.collectors.health_check.platform')
    def test_collect_success(self, mock_platform, mock_getenv, mock_session_maker):
        """Test successful health check collection."""
        # Mock platform information
        mock_platform.system.return_value = 'Linux'
        mock_platform.release.return_value = '5.4.0'
        mock_platform.python_version.return_value = '3.11.0'

        # Mock environment variables
        mock_getenv.side_effect = lambda key: {
            'GITHUB_APP_CLIENT_ID': 'test_client_id',
            'KEYCLOAK_SERVER_URL': 'https://keycloak.example.com',
        }.get(key)

        # Mock database health check
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session

        results = self.collector.collect()

        # Verify we got expected metrics
        assert len(results) >= 7

        result_dict = {r.key: r.value for r in results}
        assert 'collection_timestamp' in result_dict
        assert result_dict['platform_system'] == 'Linux'
        assert result_dict['platform_release'] == '5.4.0'
        assert result_dict['python_version'] == '3.11.0'
        assert result_dict['database_healthy'] is True
        assert result_dict['has_github_app_config'] is True
        assert result_dict['has_keycloak_config'] is True
        assert 'collector_uptime_seconds' in result_dict

    @patch('enterprise.telemetry.collectors.health_check.session_maker')
    def test_database_health_check_failure(self, mock_session_maker):
        """Test database health check when database is unavailable."""
        mock_session_maker.return_value.__enter__.side_effect = Exception(
            'DB Connection Failed'
        )

        result = self.collector._check_database_health()
        assert result is False

    @patch('enterprise.telemetry.collectors.health_check.session_maker')
    def test_database_health_check_success(self, mock_session_maker):
        """Test database health check when database is healthy."""
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session

        result = self.collector._check_database_health()
        assert result is True
        mock_session.execute.assert_called_once_with('SELECT 1')

    @patch('enterprise.telemetry.collectors.health_check.session_maker')
    @patch('enterprise.telemetry.collectors.health_check.platform')
    def test_collect_with_partial_failure(self, mock_platform, mock_session_maker):
        """Test collection when some metrics fail but others succeed."""
        # Mock platform to raise an exception
        mock_platform.system.side_effect = Exception('Platform error')

        # Mock database to work
        mock_session = MagicMock()
        mock_session_maker.return_value.__enter__.return_value = mock_session

        results = self.collector.collect()

        # Should still return some results, including error metric
        assert len(results) > 0
        result_dict = {r.key: r.value for r in results}
        assert 'health_check_error' in result_dict

    def test_uptime_tracking(self):
        """Test that uptime is tracked across multiple collections."""
        # First collection should initialize start time
        results1 = self.collector.collect()
        result_dict1 = {r.key: r.value for r in results1}
        uptime1 = result_dict1.get('collector_uptime_seconds', 0)

        # Second collection should have same or higher uptime
        results2 = self.collector.collect()
        result_dict2 = {r.key: r.value for r in results2}
        uptime2 = result_dict2.get('collector_uptime_seconds', 0)

        assert uptime2 >= uptime1
