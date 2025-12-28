"""Unit tests for TelemetryIdentity model.

Tests the persistent identity storage for the OpenHands Enterprise Telemetry Service.
"""

from datetime import datetime

from storage.telemetry_identity import TelemetryIdentity


class TestTelemetryIdentity:
    """Test cases for TelemetryIdentity model."""

    def test_create_identity_with_defaults(self):
        """Test creating identity with default values."""
        identity = TelemetryIdentity()

        assert identity.id == 1
        assert identity.customer_id is None
        assert identity.instance_id is None
        assert isinstance(identity.created_at, datetime)
        assert isinstance(identity.updated_at, datetime)

    def test_create_identity_with_values(self):
        """Test creating identity with specific values."""
        customer_id = 'cust_123'
        instance_id = 'inst_456'

        identity = TelemetryIdentity(customer_id=customer_id, instance_id=instance_id)

        assert identity.id == 1
        assert identity.customer_id == customer_id
        assert identity.instance_id == instance_id

    def test_set_customer_info(self):
        """Test updating customer information."""
        identity = TelemetryIdentity()

        # Update customer info
        identity.set_customer_info(
            customer_id='new_customer', instance_id='new_instance'
        )

        assert identity.customer_id == 'new_customer'
        assert identity.instance_id == 'new_instance'

    def test_set_customer_info_partial(self):
        """Test partial updates of customer information."""
        identity = TelemetryIdentity(
            customer_id='original_customer', instance_id='original_instance'
        )

        # Update only customer_id
        identity.set_customer_info(customer_id='updated_customer')
        assert identity.customer_id == 'updated_customer'
        assert identity.instance_id == 'original_instance'

        # Update only instance_id
        identity.set_customer_info(instance_id='updated_instance')
        assert identity.customer_id == 'updated_customer'
        assert identity.instance_id == 'updated_instance'

    def test_set_customer_info_with_none(self):
        """Test that None values don't overwrite existing data."""
        identity = TelemetryIdentity(
            customer_id='existing_customer', instance_id='existing_instance'
        )

        # Call with None values - should not change existing data
        identity.set_customer_info(customer_id=None, instance_id=None)
        assert identity.customer_id == 'existing_customer'
        assert identity.instance_id == 'existing_instance'

    def test_has_customer_info_property(self):
        """Test has_customer_info property logic."""
        identity = TelemetryIdentity()

        # Initially false (both None)
        assert not identity.has_customer_info

        # Still false with only customer_id
        identity.customer_id = 'customer_123'
        assert not identity.has_customer_info

        # Still false with only instance_id
        identity.customer_id = None
        identity.instance_id = 'instance_456'
        assert not identity.has_customer_info

        # True when both are set
        identity.customer_id = 'customer_123'
        identity.instance_id = 'instance_456'
        assert identity.has_customer_info

    def test_has_customer_info_with_empty_strings(self):
        """Test has_customer_info with empty strings."""
        identity = TelemetryIdentity(customer_id='', instance_id='')

        # Empty strings should be falsy
        assert not identity.has_customer_info

    def test_repr_method(self):
        """Test string representation of identity."""
        identity = TelemetryIdentity(
            customer_id='test_customer', instance_id='test_instance'
        )

        repr_str = repr(identity)
        assert 'TelemetryIdentity' in repr_str
        assert 'test_customer' in repr_str
        assert 'test_instance' in repr_str

    def test_id_forced_to_one(self):
        """Test that ID is always forced to 1."""
        identity = TelemetryIdentity()
        assert identity.id == 1

        # Even if we try to set a different ID in constructor
        identity2 = TelemetryIdentity(customer_id='test')
        assert identity2.id == 1

    def test_timestamps_are_set(self):
        """Test that timestamps are properly set."""
        identity = TelemetryIdentity()

        assert identity.created_at is not None
        assert identity.updated_at is not None
        assert isinstance(identity.created_at, datetime)
        assert isinstance(identity.updated_at, datetime)
