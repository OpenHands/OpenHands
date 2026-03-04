"""Store for managing experiment assignments.

This store handles creating and updating experiment assignments for conversations.
"""

from sqlalchemy.dialects.postgresql import insert
from storage.database import session_maker
from storage.experiment_assignment import ExperimentAssignment

from openhands.core.logger import openhands_logger as logger


class ExperimentAssignmentStore:
    """Store for managing experiment assignments."""

    def update_experiment_variant(
        self,
        conversation_id: str,
        experiment_name: str,
        variant: str,
    ) -> None:
        """Update the variant for a specific experiment.

        Args:
            conversation_id: The conversation ID
            experiment_name: The name of the experiment
            variant: The variant assigned
        """
        with session_maker() as session:
            # Use PostgreSQL's INSERT ... ON CONFLICT DO NOTHING to handle unique constraint
            stmt = insert(ExperimentAssignment).values(
                conversation_id=conversation_id,
                experiment_name=experiment_name,
                variant=variant,
            )
            stmt = stmt.on_conflict_do_nothing(
                constraint='uq_experiment_assignments_conversation_experiment'
            )

            session.execute(stmt)
            session.commit()

            logger.info(
                'experiment_assignment_store:upserted_variant',
                extra={
                    'conversation_id': conversation_id,
                    'experiment_name': experiment_name,
                    'variant': variant,
                },
            )
