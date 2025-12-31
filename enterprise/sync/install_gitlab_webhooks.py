import asyncio
from typing import cast
from uuid import uuid4

from integrations.types import GitLabResourceType
from integrations.utils import GITLAB_WEBHOOK_URL
from sqlalchemy import text
from storage.database import a_session_maker
from storage.gitlab_webhook import GitlabWebhook, WebhookStatus
from storage.gitlab_webhook_store import GitlabWebhookStore

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.gitlab.gitlab_service import GitLabServiceImpl
from openhands.integrations.service_types import GitService

CHUNK_SIZE = 100
WEBHOOK_NAME = 'OpenHands Resolver'
SCOPES: list[str] = [
    'note_events',
    'merge_requests_events',
    'confidential_issues_events',
    'issues_events',
    'confidential_note_events',
    'job_events',
    'pipeline_events',
]


class BreakLoopException(Exception):
    pass


class VerifyWebhookStatus:
    async def fetch_rows(self, webhook_store: GitlabWebhookStore):
        webhooks = await webhook_store.filter_rows(limit=CHUNK_SIZE)

        return webhooks

    def determine_if_rate_limited(
        self,
        status: WebhookStatus | None,
    ) -> None:
        if status == WebhookStatus.RATE_LIMITED:
            raise BreakLoopException()

    async def check_if_resource_exists(
        self,
        gitlab_service: type[GitService],
        resource_type: GitLabResourceType,
        resource_id: str,
        webhook_store: GitlabWebhookStore,
        webhook: GitlabWebhook,
    ):
        """
        Check if the GitLab resource still exists
        """
        from integrations.gitlab.gitlab_service import SaaSGitLabService

        gitlab_service = cast(type[SaaSGitLabService], gitlab_service)

        does_resource_exist, status = await gitlab_service.check_resource_exists(
            resource_type, resource_id
        )

        logger.info(
            'Does resource exists',
            extra={
                'does_resource_exist': does_resource_exist,
                'status': status,
                'resource_id': resource_id,
                'resource_type': resource_type,
            },
        )

        self.determine_if_rate_limited(status)
        if not does_resource_exist and status != WebhookStatus.RATE_LIMITED:
            await webhook_store.delete_webhook(webhook)
            raise BreakLoopException()

    async def check_if_user_has_admin_acccess_to_resource(
        self,
        gitlab_service: type[GitService],
        resource_type: GitLabResourceType,
        resource_id: str,
        webhook_store: GitlabWebhookStore,
        webhook: GitlabWebhook,
    ):
        """
        Check is user still has permission to resource
        """
        from integrations.gitlab.gitlab_service import SaaSGitLabService

        gitlab_service = cast(type[SaaSGitLabService], gitlab_service)

        (
            is_user_admin_of_resource,
            status,
        ) = await gitlab_service.check_user_has_admin_access_to_resource(
            resource_type, resource_id
        )

        logger.info(
            'Is user admin',
            extra={
                'is_user_admin': is_user_admin_of_resource,
                'status': status,
                'resource_id': resource_id,
                'resource_type': resource_type,
            },
        )

        self.determine_if_rate_limited(status)
        if not is_user_admin_of_resource:
            await webhook_store.delete_webhook(webhook)
            raise BreakLoopException()

    async def check_if_webhook_already_exists_on_resource(
        self,
        gitlab_service: type[GitService],
        resource_type: GitLabResourceType,
        resource_id: str,
        webhook_store: GitlabWebhookStore,
        webhook: GitlabWebhook,
    ):
        """
        Check whether webhook already exists on resource
        """
        from integrations.gitlab.gitlab_service import SaaSGitLabService

        gitlab_service = cast(type[SaaSGitLabService], gitlab_service)
        (
            does_webhook_exist_on_resource,
            status,
        ) = await gitlab_service.check_webhook_exists_on_resource(
            resource_type, resource_id, GITLAB_WEBHOOK_URL
        )

        logger.info(
            'Does webhook already exist',
            extra={
                'does_webhook_exist_on_resource': does_webhook_exist_on_resource,
                'status': status,
                'resource_id': resource_id,
                'resource_type': resource_type,
            },
        )

        self.determine_if_rate_limited(status)
        if does_webhook_exist_on_resource != webhook.webhook_exists:
            await webhook_store.update_webhook(
                webhook, {'webhook_exists': does_webhook_exist_on_resource}
            )

        if does_webhook_exist_on_resource:
            raise BreakLoopException()

    async def verify_conditions_are_met(
        self,
        gitlab_service: type[GitService],
        resource_type: GitLabResourceType,
        resource_id: str,
        webhook_store: GitlabWebhookStore,
        webhook: GitlabWebhook,
    ):
        await self.check_if_resource_exists(
            gitlab_service=gitlab_service,
            resource_type=resource_type,
            resource_id=resource_id,
            webhook_store=webhook_store,
            webhook=webhook,
        )

        await self.check_if_user_has_admin_acccess_to_resource(
            gitlab_service=gitlab_service,
            resource_type=resource_type,
            resource_id=resource_id,
            webhook_store=webhook_store,
            webhook=webhook,
        )

        await self.check_if_webhook_already_exists_on_resource(
            gitlab_service=gitlab_service,
            resource_type=resource_type,
            resource_id=resource_id,
            webhook_store=webhook_store,
            webhook=webhook,
        )

    async def create_new_webhook(
        self,
        gitlab_service: type[GitService],
        resource_type: GitLabResourceType,
        resource_id: str,
        webhook_store: GitlabWebhookStore,
        webhook: GitlabWebhook,
    ):
        """
        Install webhook on resource
        """
        from integrations.gitlab.gitlab_service import SaaSGitLabService

        gitlab_service = cast(type[SaaSGitLabService], gitlab_service)

        webhook_secret = f'{webhook.user_id}-{str(uuid4())}'
        webhook_uuid = f'{str(uuid4())}'

        webhook_id, status = await gitlab_service.install_webhook(
            resource_type=resource_type,
            resource_id=resource_id,
            webhook_name=WEBHOOK_NAME,
            webhook_url=GITLAB_WEBHOOK_URL,
            webhook_secret=webhook_secret,
            webhook_uuid=webhook_uuid,
            scopes=SCOPES,
        )

        logger.info(
            'Creating new webhook',
            extra={
                'webhook_id': webhook_id,
                'status': status,
                'resource_id': resource_id,
                'resource_type': resource_type,
            },
        )

        self.determine_if_rate_limited(status)

        if webhook_id:
            await webhook_store.update_webhook(
                webhook=webhook,
                update_fields={
                    'webhook_secret': webhook_secret,
                    'webhook_exists': True,  # webhook was created
                    'webhook_url': GITLAB_WEBHOOK_URL,
                    'scopes': SCOPES,
                    'webhook_uuid': webhook_uuid,  # required to identify which webhook installation is sending payload
                },
            )

            logger.info(
                f'Installed webhook for {webhook.user_id} on {resource_type}:{resource_id}'
            )

    async def install_webhooks(self):
        """
        Periodically check the conditions for installing a webhook on resource as valid
        Rows with valid conditions with contain (webhook_exists=False, status=WebhookStatus.VERIFIED)

        Conditions we check for
            1. Resoure exists
                - user could have deleted resource
            2. User has admin access to resource
                - user's permissions to install webhook could have changed
            3. Webhook exists
                - user could have removed webhook from resource
                - resource was never setup with webhook

        """

        from integrations.gitlab.gitlab_service import SaaSGitLabService

        # Check if the table exists before proceeding
        # This handles cases where the CronJob runs before database migrations complete
        async with a_session_maker() as session:
            query = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = 'gitlab_webhook'
                )
            """)
            result = await session.execute(query)
            table_exists = result.scalar() or False

        if not table_exists:
            logger.info(
                'gitlab_webhook table does not exist yet, '
                'waiting for database migrations to complete'
            )
            return

        # Get an instance of the webhook store
        webhook_store = await GitlabWebhookStore.get_instance()

        # Load chunks of rows that need processing (webhook_exists == False)
        webhooks_to_process = await self.fetch_rows(webhook_store)

        logger.info(
            'Processing webhook chunks',
            extra={'webhooks_to_process': webhooks_to_process},
        )

        for webhook in webhooks_to_process:
            try:
                user_id = webhook.user_id
                resource_type, resource_id = GitlabWebhookStore.determine_resource_type(
                    webhook
                )

                gitlab_service_impl = GitLabServiceImpl(external_auth_id=user_id)

                if not isinstance(gitlab_service_impl, SaaSGitLabService):
                    raise Exception('Only SaaSGitLabService is supported')
                # Cast needed when mypy can see OpenHands
                gitlab_service = cast(type[SaaSGitLabService], gitlab_service_impl)

                await self.verify_conditions_are_met(
                    gitlab_service=gitlab_service,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    webhook_store=webhook_store,
                    webhook=webhook,
                )

                # Conditions have been met for installing webhook
                await self.create_new_webhook(
                    gitlab_service=gitlab_service,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    webhook_store=webhook_store,
                    webhook=webhook,
                )

            except BreakLoopException:
                pass  # Continue processing but still update last_synced
            finally:
                # Always update last_synced after processing (success or failure)
                # to prevent immediate reprocessing of the same webhook
                try:
                    await webhook_store.update_last_synced(webhook)
                except Exception as e:
                    logger.warning(
                        'Failed to update last_synced for webhook',
                        extra={
                            'webhook_id': getattr(webhook, 'id', None),
                            'project_id': getattr(webhook, 'project_id', None),
                            'group_id': getattr(webhook, 'group_id', None),
                            'error': str(e),
                        },
                    )


if __name__ == '__main__':
    status_verifier = VerifyWebhookStatus()
    asyncio.run(status_verifier.install_webhooks())
