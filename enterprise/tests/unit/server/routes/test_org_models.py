# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for enterprise.server.routes.org_models."""

from unittest.mock import MagicMock, PropertyMock
from uuid import uuid4

from server.auth.authorization import RoleName
from server.routes.org_models import MeResponse
from storage.org_member import OrgMember


class TestMeResponseFromOrgMember:
    """Tests for MeResponse.from_org_member() - regression tests for GitHub #14898."""

    def test_from_org_member_with_custom_llm_api_key(self):
        """When has_custom_llm_api_key is True, member's API key is masked and returned."""
        member = MagicMock(spec=OrgMember)
        member.org_id = uuid4()
        member.user_id = uuid4()
        member.agent_settings_diff = {}
        member.conversation_settings_diff = {}
        member.status = 'active'
        member.has_custom_llm_api_key = True
        type(member).llm_api_key = PropertyMock(return_value='sk-test-secret123')
        type(member).llm_api_key_for_byor = PropertyMock(return_value=None)

        role = MagicMock()
        role.name = RoleName.MEMBER

        result = MeResponse.from_org_member(member, role, 'test@example.com')

        assert result.llm_api_key == '****t123'
        assert result.role == RoleName.MEMBER
        assert result.email == 'test@example.com'

    def test_from_org_member_populates_permissions_from_role(self):
        """Permissions are derived from the role for server-defined client gating."""
        member = MagicMock(spec=OrgMember)
        member.org_id = uuid4()
        member.user_id = uuid4()
        member.agent_settings_diff = {}
        member.conversation_settings_diff = {}
        member.status = 'active'
        member.has_custom_llm_api_key = False
        type(member).llm_api_key = PropertyMock(return_value=None)
        type(member).llm_api_key_for_byor = PropertyMock(return_value=None)

        admin_role = MagicMock()
        admin_role.name = RoleName.ADMIN
        admin = MeResponse.from_org_member(member, admin_role, 'admin@example.com')
        assert 'edit_org_settings' in admin.permissions
        assert 'view_org_settings' in admin.permissions

        member_role = MagicMock()
        member_role.name = RoleName.MEMBER
        viewer = MeResponse.from_org_member(member, member_role, 'm@example.com')
        # Members may view but not edit; the client gates LLM-profile mutations
        # on the absence of edit_org_settings.
        assert 'edit_org_settings' not in viewer.permissions
        assert 'view_org_settings' in viewer.permissions

    def test_from_org_member_without_custom_llm_api_key_returns_empty_string(self):
        """When has_custom_llm_api_key is False, returns '' without accessing member.llm_api_key.

        This is a regression test for GitHub #14898 - when has_custom_llm_api_key is False,
        the code must NOT try to access member.llm_api_key because it will attempt
        decryption which fails on empty/null stored values.
        """
        member = MagicMock(spec=OrgMember)
        member.org_id = uuid4()
        member.user_id = uuid4()
        member.agent_settings_diff = {}
        member.conversation_settings_diff = {}
        member.status = 'active'
        member.has_custom_llm_api_key = False

        # Set llm_api_key to raise an error if accessed - this verifies we don't access it
        # when has_custom_llm_api_key is False
        def raise_on_access():
            raise AssertionError(
                'member.llm_api_key should not be accessed when has_custom_llm_api_key is False'
            )

        type(member).llm_api_key = PropertyMock(side_effect=raise_on_access)
        type(member).llm_api_key_for_byor = PropertyMock(return_value=None)

        role = MagicMock()
        role.name = RoleName.MEMBER

        result = MeResponse.from_org_member(member, role, 'test@example.com')

        # Must return empty string when has_custom_llm_api_key is False
        assert result.llm_api_key == ''
        assert result.role == RoleName.MEMBER
        assert result.email == 'test@example.com'

    def test_from_org_member_with_llm_api_key_for_byor(self):
        """BYOR key is masked and included when present."""
        member = MagicMock(spec=OrgMember)
        member.org_id = uuid4()
        member.user_id = uuid4()
        member.agent_settings_diff = {}
        member.conversation_settings_diff = {}
        member.status = 'active'
        member.has_custom_llm_api_key = True
        type(member).llm_api_key = PropertyMock(return_value='sk-member-key')
        type(member).llm_api_key_for_byor = PropertyMock(return_value='sk-byor-key')

        role = MagicMock()
        role.name = RoleName.ADMIN

        result = MeResponse.from_org_member(member, role, 'admin@example.com')

        assert result.llm_api_key == '****-key'
        assert result.llm_api_key_for_byor == '****-key'
