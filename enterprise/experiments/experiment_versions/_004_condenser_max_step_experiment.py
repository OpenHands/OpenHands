"""
Condenser max step experiment handler.

This module contains the handler for the condenser max step experiment that tests
different max_size values for the condenser configuration.
"""

from uuid import UUID

import posthog
from experiments.constants import EXPERIMENT_CONDENSER_MAX_STEP
from server.constants import IS_FEATURE_ENV
from storage.experiment_assignment_store import ExperimentAssignmentStore

from openhands.core.logger import openhands_logger as logger
from openhands.sdk import Agent
from openhands.sdk.context.condenser import (
    LLMSummarizingCondenser,
)
from openhands.server.session.conversation_init_data import ConversationInitData


def _get_condenser_max_step_variant(user_id, conversation_id):
    """
    Get the condenser max step variant for the experiment.

    Args:
        user_id: The user ID
        conversation_id: The conversation ID

    Returns:
        str or None: The PostHog variant name or None if experiment is not enabled or error occurs
    """
    # No-op if the specific experiment is not enabled
    if not EXPERIMENT_CONDENSER_MAX_STEP:
        logger.info(
            'experiment_manager_004:ab_testing:skipped',
            extra={
                'convo_id': conversation_id,
                'reason': 'experiment_not_enabled',
                'experiment': EXPERIMENT_CONDENSER_MAX_STEP,
            },
        )
        return None

    # Use experiment name as the flag key
    try:
        enabled_variant = posthog.get_feature_flag(
            EXPERIMENT_CONDENSER_MAX_STEP, conversation_id
        )
    except Exception as e:
        logger.error(
            'experiment_manager:get_feature_flag:failed',
            extra={
                'convo_id': conversation_id,
                'experiment': EXPERIMENT_CONDENSER_MAX_STEP,
                'error': str(e),
            },
        )
        return None

    # Store the experiment assignment in the database
    try:
        experiment_store = ExperimentAssignmentStore()
        experiment_store.update_experiment_variant(
            conversation_id=conversation_id,
            experiment_name='condenser_max_step_experiment',
            variant=enabled_variant,
        )
    except Exception as e:
        logger.error(
            'experiment_manager:store_assignment:failed',
            extra={
                'convo_id': conversation_id,
                'experiment': EXPERIMENT_CONDENSER_MAX_STEP,
                'variant': enabled_variant,
                'error': str(e),
            },
        )
        # Fail the experiment if we cannot track the splits - results would not be explainable
        return None

    # Log the experiment event
    # If this is a feature environment, add "FEATURE_" prefix to user_id for PostHog
    posthog_user_id = f'FEATURE_{user_id}' if IS_FEATURE_ENV else user_id

    try:
        posthog.capture(
            distinct_id=posthog_user_id,
            event='condenser_max_step_set',
            properties={
                'conversation_id': conversation_id,
                'variant': enabled_variant,
                'original_user_id': user_id,
                'is_feature_env': IS_FEATURE_ENV,
            },
        )
    except Exception as e:
        logger.error(
            'experiment_manager:posthog_capture:failed',
            extra={
                'convo_id': conversation_id,
                'experiment': EXPERIMENT_CONDENSER_MAX_STEP,
                'error': str(e),
            },
        )
        # Continue execution as this is not critical

    logger.info(
        'posthog_capture',
        extra={
            'event': 'condenser_max_step_set',
            'posthog_user_id': posthog_user_id,
            'is_feature_env': IS_FEATURE_ENV,
            'conversation_id': conversation_id,
            'variant': enabled_variant,
        },
    )

    return enabled_variant


def handle_condenser_max_step_experiment(
    user_id: str | None,
    conversation_id: str,
    conversation_settings: ConversationInitData,
) -> ConversationInitData:
    """
    Handle the condenser max step experiment for conversation settings.

    We should not modify persistent user settings. Instead, apply the experiment
    variant to the conversation's in-memory settings object for this session only.

    Variants:
    - control -> condenser_max_size = 120
    - treatment -> condenser_max_size = 80

    Returns the (potentially) modified conversation_settings.
    """

    enabled_variant = _get_condenser_max_step_variant(user_id, conversation_id)

    if enabled_variant is None:
        return conversation_settings

    if enabled_variant == 'control':
        condenser_max_size = 120
    elif enabled_variant == 'treatment':
        condenser_max_size = 80
    else:
        logger.error(
            'condenser_max_step_experiment:unknown_variant',
            extra={
                'user_id': user_id,
                'convo_id': conversation_id,
                'variant': enabled_variant,
                'reason': 'unknown variant; returning original conversation settings',
            },
        )
        return conversation_settings

    try:
        # Apply the variant to this conversation only; do not persist to DB.
        # Not all OpenHands versions expose `condenser_max_size` on settings.
        if hasattr(conversation_settings, 'condenser_max_size'):
            conversation_settings.condenser_max_size = condenser_max_size
            logger.info(
                'condenser_max_step_experiment:conversation_settings_applied',
                extra={
                    'user_id': user_id,
                    'convo_id': conversation_id,
                    'variant': enabled_variant,
                    'condenser_max_size': condenser_max_size,
                },
            )
        else:
            logger.warning(
                'condenser_max_step_experiment:field_missing_on_settings',
                extra={
                    'user_id': user_id,
                    'convo_id': conversation_id,
                    'variant': enabled_variant,
                    'reason': 'condenser_max_size not present on ConversationInitData',
                },
            )
    except Exception as e:
        logger.error(
            'condenser_max_step_experiment:apply_failed',
            extra={
                'user_id': user_id,
                'convo_id': conversation_id,
                'variant': enabled_variant,
                'error': str(e),
            },
        )
        return conversation_settings

    return conversation_settings


def handle_condenser_max_step_experiment__v1(
    user_id: str | None,
    conversation_id: UUID,
    agent: Agent,
) -> Agent:
    enabled_variant = _get_condenser_max_step_variant(user_id, str(conversation_id))

    if enabled_variant is None:
        return agent

    if enabled_variant == 'control':
        condenser_max_size = 120
    elif enabled_variant == 'treatment':
        condenser_max_size = 80
    else:
        logger.error(
            'condenser_max_step_experiment:unknown_variant',
            extra={
                'user_id': user_id,
                'convo_id': conversation_id,
                'variant': enabled_variant,
                'reason': 'unknown variant; returning original conversation settings',
            },
        )
        return agent

    condenser_llm = agent.llm.model_copy(update={'usage_id': 'condenser'})
    condenser = LLMSummarizingCondenser(
        llm=condenser_llm, max_size=condenser_max_size, keep_first=4
    )

    return agent.model_copy(update={'condenser': condenser})
