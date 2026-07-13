from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import SecretStr
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from openhands.sdk.settings import ConversationSettings, default_agent_settings

logging.getLogger('sqlalchemy').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.orm').setLevel(logging.WARNING)
logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)

MANAGED_REFRESH_USER_ID = '11111111-1111-1111-1111-111111111111'
MANAGED_REFRESH_ORG_ID = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa1'
MANAGED_START_USER_ID = '22222222-2222-2222-2222-222222222222'
MANAGED_START_ORG_ID = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa2'
BYOK_USER_ID = '33333333-3333-3333-3333-333333333333'
BYOK_ORG_ID = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa3'
NONMANAGED_USER_ID = '44444444-4444-4444-4444-444444444444'
NONMANAGED_ORG_ID = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaa4'

MANAGED_REFRESH_OLD_KEY = 'live-llm-managed-refresh-old'
MANAGED_START_OLD_KEY = 'live-llm-managed-start-old'
BYOK_LLM_KEY = 'live-llm-byok-user-key'
BYOK_BYOR_KEY = 'live-llm-byor-export-key'
NONMANAGED_LLM_KEY = 'live-llm-nonmanaged-user-key'

API_KEYS = {
    'managed_refresh': 'live-api-managed-refresh',
    'managed_start': 'live-api-managed-start',
    'byok': 'live-api-byok',
    'nonmanaged': 'live-api-nonmanaged',
}


@dataclass(frozen=True)
class ScenarioUser:
    name: str
    user_id: str
    org_id: str
    email: str
    api_key: str
    llm_key: str
    model: str
    base_url: str
    has_custom_llm_key: bool = False
    byor_key: str | None = None


def fingerprint(value: str | None) -> str | None:
    if value is None:
        return None
    import hashlib

    return hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]


def get_openhands_cloud_key_alias(user_id: str, org_id: str) -> str:
    return f'OpenHands Cloud - user {user_id} - org {org_id}'


def import_models() -> None:
    from server.verified_models.verified_model_service import (
        StoredVerifiedModel,  # noqa: F401
    )
    from storage.api_key import ApiKey  # noqa: F401
    from storage.billing_session import BillingSession  # noqa: F401
    from storage.conversation_work import ConversationWork  # noqa: F401
    from storage.device_code import DeviceCode  # noqa: F401
    from storage.feedback import Feedback  # noqa: F401
    from storage.github_app_installation import GithubAppInstallation  # noqa: F401
    from storage.org import Org  # noqa: F401
    from storage.org_budget_settings import OrgBudgetSettings  # noqa: F401
    from storage.org_budget_threshold import OrgBudgetThreshold  # noqa: F401
    from storage.org_git_claim import OrgGitClaim  # noqa: F401
    from storage.org_invitation import OrgInvitation  # noqa: F401
    from storage.org_member import OrgMember  # noqa: F401
    from storage.org_user_budget_override import OrgUserBudgetOverride  # noqa: F401
    from storage.role import Role  # noqa: F401
    from storage.slack_conversation import SlackConversation  # noqa: F401
    from storage.slack_user import SlackUser  # noqa: F401
    from storage.stored_conversation_metadata import (
        StoredConversationMetadata,  # noqa: F401
    )
    from storage.stored_conversation_metadata_saas import (
        StoredConversationMetadataSaas,  # noqa: F401
    )
    from storage.stored_offline_token import StoredOfflineToken  # noqa: F401
    from storage.stripe_customer import StripeCustomer  # noqa: F401
    from storage.user import User  # noqa: F401
    from storage.user_settings import UserSettings  # noqa: F401

    from openhands.app_server.app_conversation.sql_app_conversation_start_task_service import (  # noqa: F401,E501
        StoredAppConversationStartTask,
    )
    from openhands.app_server.event_callback.sql_event_callback_service import (  # noqa: F401,E501
        StoredEventCallback,
        StoredEventCallbackResult,
    )
    from openhands.app_server.pending_messages.pending_message_service import (  # noqa: F401,E501
        StoredPendingMessage,
    )


def create_required_tables(engine: Any) -> None:
    from storage.api_key import ApiKey
    from storage.auth_tokens import AuthTokens
    from storage.base import Base as EnterpriseBase
    from storage.jira_dc_user import JiraDcUser
    from storage.jira_dc_workspace import JiraDcWorkspace
    from storage.org import Org
    from storage.org_member import OrgMember
    from storage.role import Role
    from storage.stored_conversation_metadata_saas import StoredConversationMetadataSaas
    from storage.stored_custom_secrets import StoredCustomSecrets
    from storage.user import User
    from storage.user_settings import UserSettings

    from openhands.app_server.app_conversation.sql_app_conversation_info_service import (
        StoredConversationCostEvent,
        StoredConversationMetadata,
    )
    from openhands.app_server.app_conversation.sql_app_conversation_start_task_service import (
        StoredAppConversationStartTask,
    )
    from openhands.app_server.event_callback.sql_event_callback_service import (
        StoredEventCallback,
        StoredEventCallbackResult,
    )
    from openhands.app_server.pending_messages.pending_message_service import (
        StoredPendingMessage,
    )
    from openhands.app_server.utils.sql_utils import Base as CoreBase

    EnterpriseBase.metadata.create_all(
        engine,
        tables=[
            Role.__table__,
            AuthTokens.__table__,
            Org.__table__,
            JiraDcWorkspace.__table__,
            JiraDcUser.__table__,
            User.__table__,
            OrgMember.__table__,
            ApiKey.__table__,
            UserSettings.__table__,
            StoredConversationMetadataSaas.__table__,
            StoredCustomSecrets.__table__,
        ],
    )
    CoreBase.metadata.create_all(
        engine,
        tables=[
            StoredConversationMetadata.__table__,
            StoredConversationCostEvent.__table__,
            StoredAppConversationStartTask.__table__,
            StoredEventCallback.__table__,
            StoredEventCallbackResult.__table__,
            StoredPendingMessage.__table__,
        ],
    )


def make_agent_settings(model: str, base_url: str) -> dict[str, Any]:
    settings = default_agent_settings()
    settings = settings.model_copy(
        update={
            'llm': settings.llm.model_copy(
                update={
                    'model': model,
                    'base_url': base_url,
                    'api_key': None,
                }
            )
        }
    )
    return settings.model_dump(mode='json', context={'expose_secrets': True})


def seed(db_path: Path, litellm_url: str, output_path: Path) -> None:
    import_models()

    from storage.api_key import ApiKey
    from storage.org import Org
    from storage.org_member import OrgMember
    from storage.role import Role
    from storage.user import User

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    engine = create_engine(
        f'sqlite:///{db_path}', connect_args={'check_same_thread': False}
    )
    create_required_tables(engine)
    Session = sessionmaker(bind=engine)

    scenarios = [
        ScenarioUser(
            name='managed_refresh',
            user_id=MANAGED_REFRESH_USER_ID,
            org_id=MANAGED_REFRESH_ORG_ID,
            email='managed-refresh@example.invalid',
            api_key=API_KEYS['managed_refresh'],
            llm_key=MANAGED_REFRESH_OLD_KEY,
            model='openhands/gpt-5.5',
            base_url=litellm_url,
        ),
        ScenarioUser(
            name='managed_start',
            user_id=MANAGED_START_USER_ID,
            org_id=MANAGED_START_ORG_ID,
            email='managed-start@example.invalid',
            api_key=API_KEYS['managed_start'],
            llm_key=MANAGED_START_OLD_KEY,
            model='openhands/gpt-5.5',
            base_url=litellm_url,
        ),
        ScenarioUser(
            name='byok',
            user_id=BYOK_USER_ID,
            org_id=BYOK_ORG_ID,
            email='byok@example.invalid',
            api_key=API_KEYS['byok'],
            llm_key=BYOK_LLM_KEY,
            model='openhands/gpt-5.5',
            base_url=litellm_url,
            has_custom_llm_key=True,
            byor_key=BYOK_BYOR_KEY,
        ),
        ScenarioUser(
            name='nonmanaged',
            user_id=NONMANAGED_USER_ID,
            org_id=NONMANAGED_ORG_ID,
            email='nonmanaged@example.invalid',
            api_key=API_KEYS['nonmanaged'],
            llm_key=NONMANAGED_LLM_KEY,
            model='gpt-4o',
            base_url='https://example.invalid/nonmanaged',
        ),
    ]

    with Session() as session:
        session.add(Role(id=1, name='member', rank=10))
        for scenario in scenarios:
            org = Org(
                id=UUID(scenario.org_id),
                name=f'live-evidence-{scenario.name}',
                agent_settings=make_agent_settings(scenario.model, scenario.base_url),
                conversation_settings=ConversationSettings().model_dump(mode='json'),
                byor_export_enabled=True,
                v1_enabled=True,
            )
            user = User(
                id=UUID(scenario.user_id),
                current_org_id=UUID(scenario.org_id),
                accepted_tos=datetime.now(UTC),
                enable_sound_notifications=False,
                user_consents_to_analytics=False,
                email=scenario.email,
                email_verified=True,
                git_full_clone=False,
                onboarding_completed=True,
            )
            member = OrgMember(
                org_id=UUID(scenario.org_id),
                user_id=UUID(scenario.user_id),
                role_id=1,
                llm_api_key=SecretStr(scenario.llm_key),
                llm_api_key_for_byor=SecretStr(scenario.byor_key)
                if scenario.byor_key
                else None,
                has_custom_llm_api_key=scenario.has_custom_llm_key,
                agent_settings_diff={},
                conversation_settings_diff={},
                status='active',
            )
            api_key = ApiKey(
                key=scenario.api_key,
                user_id=scenario.user_id,
                org_id=UUID(scenario.org_id),
                name=f'live-evidence-{scenario.name}',
            )
            session.add_all([org, user, member, api_key])
        session.commit()

    initial_litellm_keys = [
        {
            'key': MANAGED_REFRESH_OLD_KEY,
            'user_id': MANAGED_REFRESH_USER_ID,
            'team_id': MANAGED_REFRESH_ORG_ID,
            'key_alias': get_openhands_cloud_key_alias(
                MANAGED_REFRESH_USER_ID, MANAGED_REFRESH_ORG_ID
            ),
            'metadata': {'type': 'openhands'},
        },
        {
            'key': MANAGED_START_OLD_KEY,
            'user_id': MANAGED_START_USER_ID,
            'team_id': MANAGED_START_ORG_ID,
            'key_alias': get_openhands_cloud_key_alias(
                MANAGED_START_USER_ID, MANAGED_START_ORG_ID
            ),
            'metadata': {'type': 'openhands'},
        },
    ]

    output = {
        'database': str(db_path),
        'scenarios': [
            {
                **asdict(scenario),
                'api_key': {'fingerprint': fingerprint(scenario.api_key)},
                'llm_key': {'fingerprint': fingerprint(scenario.llm_key)},
                'byor_key': {'fingerprint': fingerprint(scenario.byor_key)}
                if scenario.byor_key
                else None,
            }
            for scenario in scenarios
        ],
        'api_keys': {
            name: {'fingerprint': fingerprint(value)}
            for name, value in API_KEYS.items()
        },
        'initial_litellm_keys': [
            {
                'fingerprint': fingerprint(item['key']),
                'user_id': item['user_id'],
                'team_id': item['team_id'],
                'key_alias': item['key_alias'],
                'metadata': item['metadata'],
            }
            for item in initial_litellm_keys
        ],
        'initial_litellm_keys_raw': initial_litellm_keys,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + '\n')


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(
            'usage: seed_live_db.py <sqlite-db-path> <litellm-url> <output-json>'
        )
    seed(Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))


if __name__ == '__main__':
    main()
