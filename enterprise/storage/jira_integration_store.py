from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from storage.database import session_maker
from storage.jira_conversation import JiraConversation
from storage.jira_user import JiraUser
from storage.jira_workspace import JiraWorkspace

from openhands.core.logger import openhands_logger as logger


@dataclass
class JiraIntegrationStore:
    async def create_workspace(
        self,
        name: str,
        jira_cloud_id: str,
        admin_user_id: str,
        encrypted_webhook_secret: str,
        svc_acc_email: str,
        encrypted_svc_acc_api_key: str,
        status: str = 'active',
    ) -> JiraWorkspace:
        """Create a new Jira workspace with encrypted sensitive data."""

        workspace = JiraWorkspace(
            name=name.lower(),
            jira_cloud_id=jira_cloud_id,
            admin_user_id=admin_user_id,
            webhook_secret=encrypted_webhook_secret,
            svc_acc_email=svc_acc_email,
            svc_acc_api_key=encrypted_svc_acc_api_key,
            status=status,
        )

        with session_maker() as session:
            session.add(workspace)
            session.commit()
            session.refresh(workspace)

        logger.info(f'[Jira] Created workspace {workspace.name}')
        return workspace

    async def update_workspace(
        self,
        id: int,
        jira_cloud_id: Optional[str] = None,
        encrypted_webhook_secret: Optional[str] = None,
        svc_acc_email: Optional[str] = None,
        encrypted_svc_acc_api_key: Optional[str] = None,
        status: Optional[str] = None,
    ) -> JiraWorkspace:
        """Update an existing Jira workspace with encrypted sensitive data."""
        with session_maker() as session:
            # Find existing workspace by ID
            workspace = (
                session.query(JiraWorkspace).filter(JiraWorkspace.id == id).first()
            )

            if not workspace:
                raise ValueError(f'Workspace with ID "{id}" not found')

            if jira_cloud_id is not None:
                workspace.jira_cloud_id = jira_cloud_id

            if encrypted_webhook_secret is not None:
                workspace.webhook_secret = encrypted_webhook_secret

            if svc_acc_email is not None:
                workspace.svc_acc_email = svc_acc_email

            if encrypted_svc_acc_api_key is not None:
                workspace.svc_acc_api_key = encrypted_svc_acc_api_key

            if status is not None:
                workspace.status = status

            session.commit()
            session.refresh(workspace)

        logger.info(f'[Jira] Updated workspace {workspace.name}')
        return workspace

    async def create_workspace_link(
        self,
        keycloak_user_id: str,
        jira_user_id: str,
        jira_workspace_id: int,
        status: str = 'active',
    ) -> JiraUser:
        """Create a new Jira workspace link."""

        jira_user = JiraUser(
            keycloak_user_id=keycloak_user_id,
            jira_user_id=jira_user_id,
            jira_workspace_id=jira_workspace_id,
            status=status,
        )

        with session_maker() as session:
            session.add(jira_user)
            session.commit()
            session.refresh(jira_user)

        logger.info(
            f'[Jira] Created user {jira_user.id} for workspace {jira_workspace_id}'
        )
        return jira_user

    async def get_workspace_by_id(self, workspace_id: int) -> Optional[JiraWorkspace]:
        """Retrieve workspace by ID."""
        with session_maker() as session:
            return (
                session.query(JiraWorkspace)
                .filter(JiraWorkspace.id == workspace_id)
                .first()
            )

    async def get_workspace_by_name(self, workspace_name: str) -> JiraWorkspace | None:
        """Retrieve workspace by name."""
        with session_maker() as session:
            return (
                session.query(JiraWorkspace)
                .filter(JiraWorkspace.name == workspace_name.lower())
                .first()
            )

    async def get_user_by_active_workspace(
        self, keycloak_user_id: str
    ) -> Optional[JiraUser]:
        """Get Jira user by Keycloak user ID."""
        with session_maker() as session:
            return (
                session.query(JiraUser)
                .filter(
                    JiraUser.keycloak_user_id == keycloak_user_id,
                    JiraUser.status == 'active',
                )
                .first()
            )

    async def get_user_by_keycloak_id_and_workspace(
        self, keycloak_user_id: str, jira_workspace_id: int
    ) -> Optional[JiraUser]:
        """Get Jira user by Keycloak user ID and workspace ID."""
        with session_maker() as session:
            return (
                session.query(JiraUser)
                .filter(
                    JiraUser.keycloak_user_id == keycloak_user_id,
                    JiraUser.jira_workspace_id == jira_workspace_id,
                )
                .first()
            )

    async def get_active_user(
        self, jira_user_id: str, jira_workspace_id: int
    ) -> Optional[JiraUser]:
        """Get Jira user by Keycloak user ID and workspace ID."""
        with session_maker() as session:
            return (
                session.query(JiraUser)
                .filter(
                    JiraUser.jira_user_id == jira_user_id,
                    JiraUser.jira_workspace_id == jira_workspace_id,
                    JiraUser.status == 'active',
                )
                .first()
            )

    async def update_user_integration_status(
        self, keycloak_user_id: str, status: str
    ) -> JiraUser:
        """Update Jira user integration status."""
        with session_maker() as session:
            jira_user = (
                session.query(JiraUser)
                .filter(JiraUser.keycloak_user_id == keycloak_user_id)
                .first()
            )

            if not jira_user:
                raise ValueError(
                    f'Jira user not found for Keycloak ID: {keycloak_user_id}'
                )

            jira_user.status = status
            session.commit()
            session.refresh(jira_user)

            logger.info(f'[Jira] Updated user {keycloak_user_id} status to {status}')
            return jira_user

    async def deactivate_workspace(self, workspace_id: int):
        """Deactivate the workspace and all user links for a given workspace."""
        with session_maker() as session:
            users = (
                session.query(JiraUser)
                .filter(
                    JiraUser.jira_workspace_id == workspace_id,
                    JiraUser.status == 'active',
                )
                .all()
            )

            for user in users:
                user.status = 'inactive'
                session.add(user)

            workspace = (
                session.query(JiraWorkspace)
                .filter(JiraWorkspace.id == workspace_id)
                .first()
            )
            if workspace:
                workspace.status = 'inactive'
                session.add(workspace)

            session.commit()

        logger.info(f'[Jira] Deactivated all user links for workspace {workspace_id}')

    async def create_conversation(self, jira_conversation: JiraConversation) -> None:
        """Create a new Jira conversation record."""
        with session_maker() as session:
            session.add(jira_conversation)
            session.commit()

    async def get_user_conversations_by_issue_id(
        self, issue_id: str, jira_user_id: int
    ) -> JiraConversation | None:
        """Get a Jira conversation by issue ID and jira user ID."""
        with session_maker() as session:
            return (
                session.query(JiraConversation)
                .filter(
                    JiraConversation.issue_id == issue_id,
                    JiraConversation.jira_user_id == jira_user_id,
                )
                .first()
            )

    @classmethod
    def get_instance(cls) -> JiraIntegrationStore:
        """Get an instance of the JiraIntegrationStore."""
        return JiraIntegrationStore()
