from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Callable

from integrations.github.github_types import (
    WorkflowRun,
    WorkflowRunGroup,
    WorkflowRunStatus,
)
from sqlalchemy import and_, delete, select, update
from storage.database import a_session_maker
from storage.proactive_convos import ProactiveConversation

from openhands.core.logger import openhands_logger as logger
from openhands.integrations.service_types import ProviderType


@dataclass
class ProactiveConversationStore:
    def get_repo_id(self, provider: ProviderType, repo_id):
        return f'{provider.value}##{repo_id}'

    async def store_workflow_information(
        self,
        provider: ProviderType,
        repo_id: str,
        incoming_commit: str,
        workflow: WorkflowRun,
        pr_number: int,
        get_all_workflows: Callable,
    ) -> WorkflowRunGroup | None:
        """
        1. Get the workflow based on repo_id, pr_number, commit
        2. If the field doesn't exist
            - Fetch the workflow statuses and store them
            - Create a new record
        3. Check the incoming workflow run payload, and update statuses based on its fields
        4. If all statuses are completed with at least one failure, return WorkflowGroup information else None

        This method uses an explicit transaction with row-level locking to ensure
        thread safety when multiple processes access the same database rows.
        """

        should_send = False
        provider_repo_id = self.get_repo_id(provider, repo_id)

        final_workflow_group = None

        async with a_session_maker() as session:
            # Start an explicit transaction with row-level locking
            async with session.begin():
                # Get the existing proactive conversation entry with FOR UPDATE lock
                # This ensures exclusive access to these rows during the transaction
                stmt = (
                    select(ProactiveConversation)
                    .where(
                        and_(
                            ProactiveConversation.repo_id == provider_repo_id,
                            ProactiveConversation.pr_number == pr_number,
                            ProactiveConversation.commit == incoming_commit,
                        )
                    )
                    .with_for_update()  # This adds the row-level lock
                )
                result = await session.execute(stmt)
                commit_entry = result.scalars().first()

                # Interaction is complete, do not duplicate event
                if commit_entry and commit_entry.conversation_starter_sent:
                    return None

                # Get current workflow statuses
                workflow_runs = (
                    get_all_workflows()
                    if not commit_entry
                    else commit_entry.workflow_runs
                )

                workflow_run_group = (
                    workflow_runs
                    if isinstance(workflow_runs, WorkflowRunGroup)
                    else WorkflowRunGroup(**workflow_runs)
                )

                # Update with latest incoming workflow information
                workflow_run_group.runs[workflow.id] = workflow

                statuses = [
                    workflow.status for _, workflow in workflow_run_group.runs.items()
                ]

                is_none_pending = all(
                    status != WorkflowRunStatus.PENDING for status in statuses
                )

                if is_none_pending:
                    should_send = WorkflowRunStatus.FAILURE in statuses

                if should_send:
                    final_workflow_group = workflow_run_group

                if commit_entry:
                    # Update existing entry (either with workflow status updates, or marking as comment sent)
                    await session.execute(
                        update(ProactiveConversation)
                        .where(
                            ProactiveConversation.repo_id == provider_repo_id,
                            ProactiveConversation.pr_number == pr_number,
                            ProactiveConversation.commit == incoming_commit,
                        )
                        .values(
                            workflow_runs=workflow_run_group.model_dump(),
                            conversation_starter_sent=should_send,
                        )
                    )
                else:
                    convo_record = ProactiveConversation(
                        repo_id=provider_repo_id,
                        pr_number=pr_number,
                        commit=incoming_commit,
                        workflow_runs=workflow_run_group.model_dump(),
                        conversation_starter_sent=should_send,
                    )
                    session.add(convo_record)

        return final_workflow_group

    async def clean_old_convos(self, older_than_minutes=30):
        """
        Clean up proactive conversation records that are older than the specified time.

        Args:
            older_than_minutes: Number of minutes. Records older than this will be deleted.
                                Defaults to 30 minutes.
        """

        # Calculate the cutoff time (current time - older_than_minutes)
        cutoff_time = datetime.now(UTC) - timedelta(minutes=older_than_minutes)

        async with a_session_maker() as session:
            async with session.begin():
                # Delete records older than the cutoff time
                delete_stmt = delete(ProactiveConversation).where(
                    ProactiveConversation.last_updated_at < cutoff_time
                )
                result = await session.execute(delete_stmt)

                # Log the number of deleted records
                deleted_count = result.rowcount
                logger.info(
                    f'Deleted {deleted_count} proactive conversation records older than {older_than_minutes} minutes'
                )

    @classmethod
    async def get_instance(cls) -> ProactiveConversationStore:
        """Get an instance of the ProactiveConversationStore.

        Returns:
            An instance of ProactiveConversationStore
        """
        return ProactiveConversationStore()
