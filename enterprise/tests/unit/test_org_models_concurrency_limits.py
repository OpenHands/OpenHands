"""Tests for org_models concurrency limit fields.

This module tests the Pydantic models for concurrency limits including:
- OrgResponse max_concurrent_sandboxes field
- OrgUpdate max_concurrent_sandboxes field validation
- OrgMemberResponse concurrency fields
- OrgMemberUpdate max_concurrent_sandboxes_override validation
- OrgAppSettingsResponse and OrgAppSettingsUpdate
"""

import uuid
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError
from server.constants import (
    DEFAULT_COMMERCIAL_ORG_CONCURRENT_SANDBOXES,
    DEFAULT_PERSONAL_ORG_CONCURRENT_SANDBOXES,
)
from server.routes.org_models import (
    OrgAppSettingsResponse,
    OrgAppSettingsUpdate,
    OrgMemberUpdate,
    OrgResponse,
    OrgUpdate,
)
from storage.org import Org


class TestOrgResponseConcurrencyLimits:
    """Test cases for OrgResponse concurrency limit fields."""

    def test_org_response_includes_max_concurrent_sandboxes(self):
        """Test that OrgResponse includes max_concurrent_sandboxes field."""
        response = OrgResponse(
            id='test-org-id',
            name='Test Org',
            contact_name='Test Contact',
            contact_email='test@example.com',
            max_concurrent_sandboxes=5,
        )

        assert response.max_concurrent_sandboxes == 5

    def test_org_response_default_max_concurrent_sandboxes(self):
        """Test that OrgResponse has default for max_concurrent_sandboxes."""
        response = OrgResponse(
            id='test-org-id',
            name='Test Org',
            contact_name='Test Contact',
            contact_email='test@example.com',
        )

        assert (
            response.max_concurrent_sandboxes
            == DEFAULT_PERSONAL_ORG_CONCURRENT_SANDBOXES
        )

    def test_org_response_from_org_includes_max_concurrent_sandboxes(self):
        """Test that OrgResponse.from_org includes max_concurrent_sandboxes."""
        mock_org = MagicMock(spec=Org)
        mock_org.id = uuid.uuid4()
        mock_org.name = 'Test Org'
        mock_org.contact_name = 'Test Contact'
        mock_org.contact_email = 'test@example.com'
        mock_org.conversation_expiration = None
        mock_org.remote_runtime_resource_factor = None
        mock_org.billing_margin = None
        mock_org.enable_proactive_conversation_starters = True
        mock_org.sandbox_base_container_image = None
        mock_org.sandbox_runtime_container_image = None
        mock_org.org_version = 1
        mock_org.agent_settings = {}
        mock_org.conversation_settings = {}
        mock_org.max_budget_per_task = None
        mock_org.enable_solvability_analysis = None
        mock_org.v1_enabled = None
        mock_org.max_concurrent_sandboxes = 8

        response = OrgResponse.from_org(mock_org)

        assert response.max_concurrent_sandboxes == 8

    def test_org_response_from_org_uses_personal_default_when_none(self):
        """Test that OrgResponse.from_org uses 3 for personal orgs when None."""
        mock_org = MagicMock(spec=Org)
        mock_org.id = uuid.uuid4()
        mock_org.name = 'Test Org'
        mock_org.contact_name = 'Test Contact'
        mock_org.contact_email = 'test@example.com'
        mock_org.conversation_expiration = None
        mock_org.remote_runtime_resource_factor = None
        mock_org.billing_margin = None
        mock_org.enable_proactive_conversation_starters = True
        mock_org.sandbox_base_container_image = None
        mock_org.sandbox_runtime_container_image = None
        mock_org.org_version = 1
        mock_org.agent_settings = {}
        mock_org.conversation_settings = {}
        mock_org.max_budget_per_task = None
        mock_org.enable_solvability_analysis = None
        mock_org.v1_enabled = None
        mock_org.max_concurrent_sandboxes = None

        # Pass user_id matching org.id to simulate personal org
        response = OrgResponse.from_org(mock_org, user_id=str(mock_org.id))

        assert (
            response.max_concurrent_sandboxes
            == DEFAULT_PERSONAL_ORG_CONCURRENT_SANDBOXES
        )

    def test_org_response_from_org_uses_commercial_default_when_none(self):
        """Test that OrgResponse.from_org uses commercial default for commercial orgs when None."""
        mock_org = MagicMock(spec=Org)
        mock_org.id = uuid.uuid4()
        mock_org.name = 'Test Org'
        mock_org.contact_name = 'Test Contact'
        mock_org.contact_email = 'test@example.com'
        mock_org.conversation_expiration = None
        mock_org.remote_runtime_resource_factor = None
        mock_org.billing_margin = None
        mock_org.enable_proactive_conversation_starters = True
        mock_org.sandbox_base_container_image = None
        mock_org.sandbox_runtime_container_image = None
        mock_org.org_version = 1
        mock_org.agent_settings = {}
        mock_org.conversation_settings = {}
        mock_org.max_budget_per_task = None
        mock_org.enable_solvability_analysis = None
        mock_org.v1_enabled = None
        mock_org.max_concurrent_sandboxes = None

        # No user_id = commercial org
        response = OrgResponse.from_org(mock_org)

        assert (
            response.max_concurrent_sandboxes
            == DEFAULT_COMMERCIAL_ORG_CONCURRENT_SANDBOXES
        )


class TestOrgUpdateConcurrencyLimits:
    """Test cases for OrgUpdate concurrency limit validation."""

    def test_org_update_accepts_valid_max_concurrent_sandboxes(self):
        """Test that OrgUpdate accepts valid max_concurrent_sandboxes values."""
        update = OrgUpdate(max_concurrent_sandboxes=10)

        assert update.max_concurrent_sandboxes == 10

    def test_org_update_accepts_none_max_concurrent_sandboxes(self):
        """Test that OrgUpdate accepts None (no change)."""
        update = OrgUpdate(max_concurrent_sandboxes=None)

        assert update.max_concurrent_sandboxes is None

    def test_org_update_rejects_zero_max_concurrent_sandboxes(self):
        """Test that OrgUpdate rejects 0 value."""
        with pytest.raises(ValidationError) as exc_info:
            OrgUpdate(max_concurrent_sandboxes=0)

        assert 'max_concurrent_sandboxes' in str(exc_info.value)

    def test_org_update_rejects_negative_max_concurrent_sandboxes(self):
        """Test that OrgUpdate rejects negative values."""
        with pytest.raises(ValidationError) as exc_info:
            OrgUpdate(max_concurrent_sandboxes=-5)

        assert 'max_concurrent_sandboxes' in str(exc_info.value)

    def test_org_update_rejects_over_100_max_concurrent_sandboxes(self):
        """Test that OrgUpdate rejects values over 100."""
        with pytest.raises(ValidationError) as exc_info:
            OrgUpdate(max_concurrent_sandboxes=101)

        assert 'max_concurrent_sandboxes' in str(exc_info.value)

    def test_org_update_accepts_boundary_values(self):
        """Test that OrgUpdate accepts boundary values 1 and 100."""
        update_min = OrgUpdate(max_concurrent_sandboxes=1)
        update_max = OrgUpdate(max_concurrent_sandboxes=100)

        assert update_min.max_concurrent_sandboxes == 1
        assert update_max.max_concurrent_sandboxes == 100


class TestOrgMemberUpdateConcurrencyLimits:
    """Test cases for OrgMemberUpdate concurrency limit validation."""

    def test_org_member_update_accepts_valid_override(self):
        """Test that OrgMemberUpdate accepts valid override values."""
        update = OrgMemberUpdate(max_concurrent_sandboxes_override=15)

        assert update.max_concurrent_sandboxes_override == 15

    def test_org_member_update_accepts_none_override(self):
        """Test that OrgMemberUpdate accepts None (no change)."""
        update = OrgMemberUpdate(max_concurrent_sandboxes_override=None)

        assert update.max_concurrent_sandboxes_override is None

    def test_org_member_update_rejects_zero_override(self):
        """Test that OrgMemberUpdate rejects 0 value."""
        with pytest.raises(ValidationError) as exc_info:
            OrgMemberUpdate(max_concurrent_sandboxes_override=0)

        assert 'max_concurrent_sandboxes_override' in str(exc_info.value)

    def test_org_member_update_rejects_negative_override(self):
        """Test that OrgMemberUpdate rejects negative values."""
        with pytest.raises(ValidationError) as exc_info:
            OrgMemberUpdate(max_concurrent_sandboxes_override=-1)

        assert 'max_concurrent_sandboxes_override' in str(exc_info.value)

    def test_org_member_update_rejects_over_100_override(self):
        """Test that OrgMemberUpdate rejects values over 100."""
        with pytest.raises(ValidationError) as exc_info:
            OrgMemberUpdate(max_concurrent_sandboxes_override=101)

        assert 'max_concurrent_sandboxes_override' in str(exc_info.value)

    def test_org_member_update_can_set_both_role_and_override(self):
        """Test that OrgMemberUpdate can set both role and override."""
        update = OrgMemberUpdate(
            role='admin',
            max_concurrent_sandboxes_override=20,
        )

        assert update.role == 'admin'
        assert update.max_concurrent_sandboxes_override == 20


class TestOrgAppSettingsResponseConcurrencyLimits:
    """Test cases for OrgAppSettingsResponse concurrency limit fields."""

    def test_org_app_settings_response_includes_max_concurrent_sandboxes(self):
        """Test that OrgAppSettingsResponse includes max_concurrent_sandboxes."""
        response = OrgAppSettingsResponse(max_concurrent_sandboxes=7)

        assert response.max_concurrent_sandboxes == 7

    def test_org_app_settings_response_default_value(self):
        """Test that OrgAppSettingsResponse has default value."""
        response = OrgAppSettingsResponse()

        assert (
            response.max_concurrent_sandboxes
            == DEFAULT_PERSONAL_ORG_CONCURRENT_SANDBOXES
        )

    def test_org_app_settings_response_from_org(self):
        """Test that OrgAppSettingsResponse.from_org includes max_concurrent_sandboxes."""
        mock_org = MagicMock(spec=Org)
        mock_org.enable_proactive_conversation_starters = True
        mock_org.enable_solvability_analysis = False
        mock_org.max_budget_per_task = 10.0
        mock_org.max_concurrent_sandboxes = 12

        response = OrgAppSettingsResponse.from_org(mock_org)

        assert response.max_concurrent_sandboxes == 12


class TestOrgAppSettingsUpdateConcurrencyLimits:
    """Test cases for OrgAppSettingsUpdate concurrency limit validation."""

    def test_org_app_settings_update_accepts_valid_value(self):
        """Test that OrgAppSettingsUpdate accepts valid max_concurrent_sandboxes."""
        update = OrgAppSettingsUpdate(max_concurrent_sandboxes=8)

        assert update.max_concurrent_sandboxes == 8

    def test_org_app_settings_update_rejects_zero(self):
        """Test that OrgAppSettingsUpdate rejects 0 value."""
        with pytest.raises(ValidationError) as exc_info:
            OrgAppSettingsUpdate(max_concurrent_sandboxes=0)

        assert 'max_concurrent_sandboxes' in str(exc_info.value)

    def test_org_app_settings_update_rejects_over_100(self):
        """Test that OrgAppSettingsUpdate rejects values over 100."""
        with pytest.raises(ValidationError) as exc_info:
            OrgAppSettingsUpdate(max_concurrent_sandboxes=101)

        assert 'max_concurrent_sandboxes' in str(exc_info.value)
