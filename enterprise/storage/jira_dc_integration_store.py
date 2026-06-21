from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import UUID

from sqlalchemy import select, update
from storage.database import a_session_maker
from storage.jira_dc_conversation import JiraDcConversation
from storage.jira_dc_user import JiraDcUser
from storage.jira_dc_workspace import JiraDcWorkspace

from openhands.app_server.utils.logger import openhands_logger as logger


@dataclass
class JiraDcIntegrationStore:
    async def create_workspace(
        self,
        name: str,
        admin_user_id: str,
        org_id: UUID | None,
        encrypted_webhook_secret: str,
        svc_acc_email: str,
        encrypted_svc_acc_api_key: str,
        status: str = 'active',
    ) -> JiraDcWorkspace:
        """Create a new Jira DC workspace with encrypted sensitive data."""
        async with a_session_maker() as session:
            workspace = JiraDcWorkspace(
                name=name.lower(),
                admin_user_id=admin_user_id,
                org_id=org_id,
                webhook_secret=encrypted_webhook_secret,
                svc_acc_email=svc_acc_email,
                svc_acc_api_key=encrypted_svc_acc_api_key,
                status=status,
            )
            session.add(workspace)
            await session.commit()
            await session.refresh(workspace)
        logger.info(f'[Jira DC] Created workspace {workspace.name}')
        return workspace

    async def update_workspace(
        self,
        id: int,
        org_id: UUID | None = None,
        encrypted_webhook_secret: Optional[str] = None,
        svc_acc_email: Optional[str] = None,
        encrypted_svc_acc_api_key: Optional[str] = None,
        status: Optional[str] = None,
    ) -> JiraDcWorkspace:
        """Update an existing Jira DC workspace with encrypted sensitive data."""
        async with a_session_maker() as session:
            # Find existing workspace by ID
            result = await session.execute(
                select(JiraDcWorkspace).where(JiraDcWorkspace.id == id)
            )
            workspace = result.scalar_one_or_none()

            if not workspace:
                raise ValueError(f'Workspace with ID "{id}" not found')

            if encrypted_webhook_secret is not None:
                workspace.webhook_secret = encrypted_webhook_secret

            if org_id is not None:
                workspace.org_id = org_id

            if svc_acc_email is not None:
                workspace.svc_acc_email = svc_acc_email

            if encrypted_svc_acc_api_key is not None:
                workspace.svc_acc_api_key = encrypted_svc_acc_api_key

            if status is not None:
                workspace.status = status

            await session.commit()
            await session.refresh(workspace)

        logger.info(f'[Jira DC] Updated workspace {workspace.name}')
        return workspace

    async def create_workspace_link(
        self,
        keycloak_user_id: str,
        jira_dc_user_id: str,
        jira_dc_workspace_id: int,
        status: str = 'active',
    ) -> JiraDcUser:
        """Create a new Jira DC workspace link."""
        jira_dc_user = JiraDcUser(
            keycloak_user_id=keycloak_user_id,
            jira_dc_user_id=jira_dc_user_id,
            jira_dc_workspace_id=jira_dc_workspace_id,
            status=status,
        )

        async with a_session_maker() as session:
            session.add(jira_dc_user)
            await session.commit()
            await session.refresh(jira_dc_user)

        logger.info(
            f'[Jira DC] Created user {jira_dc_user.id} for workspace {jira_dc_workspace_id}'
        )
        return jira_dc_user

    async def get_workspace_by_id(self, workspace_id: int) -> Optional[JiraDcWorkspace]:
        """Retrieve workspace by ID."""
        async with a_session_maker() as session:
            result = await session.execute(
                select(JiraDcWorkspace).where(JiraDcWorkspace.id == workspace_id)
            )
            return result.scalar_one_or_none()

    async def get_workspace_by_name(
        self, workspace_name: str
    ) -> Optional[JiraDcWorkspace]:
        """Retrieve workspace by name."""
        async with a_session_maker() as session:
            result = await session.execute(
                select(JiraDcWorkspace).where(
                    JiraDcWorkspace.name == workspace_name.lower()
                )
            )
            return result.scalar_one_or_none()

    async def get_user_by_active_workspace(
        self, keycloak_user_id: str
    ) -> Optional[JiraDcUser]:
        """Retrieve user by Keycloak user ID."""
        async with a_session_maker() as session:
            result = await session.execute(
                select(JiraDcUser).where(
                    JiraDcUser.keycloak_user_id == keycloak_user_id,
                    JiraDcUser.status == 'active',
                )
            )
            return result.scalar_one_or_none()

    async def get_user_by_keycloak_id_and_workspace(
        self, keycloak_user_id: str, jira_dc_workspace_id: int
    ) -> Optional[JiraDcUser]:
        """Get Jira DC user by Keycloak user ID and workspace ID."""
        async with a_session_maker() as session:
            result = await session.execute(
                select(JiraDcUser).where(
                    JiraDcUser.keycloak_user_id == keycloak_user_id,
                    JiraDcUser.jira_dc_workspace_id == jira_dc_workspace_id,
                )
            )
            return result.scalar_one_or_none()

    async def get_active_user(
        self, jira_dc_user_id: str, jira_dc_workspace_id: int
    ) -> Optional[JiraDcUser]:
        """Get Jira DC user by Keycloak user ID and workspace ID."""
        async with a_session_maker() as session:
            result = await session.execute(
                select(JiraDcUser).where(
                    JiraDcUser.jira_dc_user_id == jira_dc_user_id,
                    JiraDcUser.jira_dc_workspace_id == jira_dc_workspace_id,
                    JiraDcUser.status == 'active',
                )
            )
            return result.scalar_one_or_none()

    async def get_active_user_by_keycloak_id_and_workspace(
        self, keycloak_user_id: str, jira_dc_workspace_id: int
    ) -> Optional[JiraDcUser]:
        """Get Jira DC user by Keycloak user ID and workspace ID."""
        async with a_session_maker() as session:
            result = await session.execute(
                select(JiraDcUser).where(
                    JiraDcUser.keycloak_user_id == keycloak_user_id,
                    JiraDcUser.jira_dc_workspace_id == jira_dc_workspace_id,
                    JiraDcUser.status == 'active',
                )
            )
            return result.scalar_one_or_none()

    async def update_user_integration_status(
        self, keycloak_user_id: str, jira_dc_workspace_id: int, status: str
    ) -> JiraDcUser:
        """Update the status of a Jira DC user mapping."""
        async with a_session_maker() as session:
            result = await session.execute(
                select(JiraDcUser).where(
                    JiraDcUser.keycloak_user_id == keycloak_user_id,
                    JiraDcUser.jira_dc_workspace_id == jira_dc_workspace_id,
                )
            )
            user = result.scalar_one_or_none()

            if not user:
                raise ValueError(
                    f"User with keycloak_user_id '{keycloak_user_id}' and "
                    f"jira_dc_workspace_id '{jira_dc_workspace_id}' not found"
                )

            user.status = status
            await session.commit()
            await session.refresh(user)
            logger.info(f'[Jira DC] Updated user {keycloak_user_id} status to {status}')
            return user

    async def deactivate_user_links_except_workspace(
        self, keycloak_user_id: str, jira_dc_workspace_id: int
    ) -> int:
        """Deactivate active Jira DC links for this user except the target workspace."""
        async with a_session_maker() as session:
            result = await session.execute(
                update(JiraDcUser)
                .where(
                    JiraDcUser.keycloak_user_id == keycloak_user_id,
                    JiraDcUser.jira_dc_workspace_id != jira_dc_workspace_id,
                    JiraDcUser.status == 'active',
                )
                .values(status='inactive')
            )
            await session.commit()

        deactivated_count = result.rowcount or 0
        if deactivated_count:
            logger.info(
                '[Jira DC] Deactivated %s stale active user links for user %s',
                deactivated_count,
                keycloak_user_id,
            )
        return deactivated_count

    async def deactivate_workspace(self, workspace_id: int):
        """Deactivate the workspace and all user links for a given workspace."""
        async with a_session_maker() as session:
            result = await session.execute(
                select(JiraDcUser).where(
                    JiraDcUser.jira_dc_workspace_id == workspace_id,
                    JiraDcUser.status == 'active',
                )
            )
            users = result.scalars().all()

            for user in users:
                user.status = 'inactive'
                session.add(user)

            result = await session.execute(
                select(JiraDcWorkspace).where(JiraDcWorkspace.id == workspace_id)
            )
            workspace = result.scalar_one_or_none()
            if workspace:
                workspace.status = 'inactive'
                session.add(workspace)

            await session.commit()

        logger.info(
            f'[Jira DC] Deactivated all user links for workspace {workspace_id}'
        )

    async def update_user_oauth_tokens(
        self,
        *,
        keycloak_user_id: str,
        workspace_id: int,
        encrypted_access_token: str,
        encrypted_refresh_token: str | None,
        access_token_expires_at: int,
        refresh_token_expires_at: int,
    ) -> int:
        """Persist updated OAuth tokens on the user's active workspace link."""
        async with a_session_maker() as session:
            async with session.begin():
                result = await session.execute(
                    update(JiraDcUser)
                    .where(
                        JiraDcUser.keycloak_user_id == keycloak_user_id,
                        JiraDcUser.jira_dc_workspace_id == workspace_id,
                        JiraDcUser.status == 'active',
                    )
                    .values(
                        oauth_access_token_encrypted=encrypted_access_token,
                        oauth_refresh_token_encrypted=encrypted_refresh_token,
                        oauth_access_token_expires_at=access_token_expires_at,
                        oauth_refresh_token_expires_at=refresh_token_expires_at,
                    )
                )
                return result.rowcount or 0

    async def get_user_oauth_tokens(
        self,
        *,
        keycloak_user_id: str,
        workspace_id: int,
    ) -> tuple[str, str | None, int, int] | None:
        """Return (enc_access, enc_refresh, access_expires_at, refresh_expires_at) or None."""
        async with a_session_maker() as session:
            result = await session.execute(
                select(JiraDcUser).where(
                    JiraDcUser.keycloak_user_id == keycloak_user_id,
                    JiraDcUser.jira_dc_workspace_id == workspace_id,
                    JiraDcUser.status == 'active',
                )
            )
            row = result.scalar_one_or_none()
            if not row or not row.oauth_access_token_encrypted:
                return None
            return (
                row.oauth_access_token_encrypted,
                row.oauth_refresh_token_encrypted,
                row.oauth_access_token_expires_at or 0,
                row.oauth_refresh_token_expires_at or 0,
            )

    async def create_conversation(
        self, jira_dc_conversation: JiraDcConversation
    ) -> None:
        """Create a new Jira DC conversation record."""
        async with a_session_maker() as session:
            session.add(jira_dc_conversation)
            await session.commit()

    async def get_user_conversations_by_issue_id(
        self, issue_id: str, jira_dc_user_id: int
    ) -> JiraDcConversation | None:
        """Get a Jira DC conversation by issue ID and jira dc user ID."""
        async with a_session_maker() as session:
            result = await session.execute(
                select(JiraDcConversation).where(
                    JiraDcConversation.issue_id == issue_id,
                    JiraDcConversation.jira_dc_user_id == jira_dc_user_id,
                )
            )
            return result.scalar_one_or_none()

    @classmethod
    def get_instance(cls) -> JiraDcIntegrationStore:
        """Get an instance of the JiraDcIntegrationStore."""
        return JiraDcIntegrationStore()
