"""System prompt experiment handler.

This module contains the handler for the system prompt experiment that uses
the PostHog variant as the system prompt filename.
"""

import copy

import posthog
from experiments.constants import EXPERIMENT_SYSTEM_PROMPT_EXPERIMENT
from server.constants import IS_FEATURE_ENV
from storage.experiment_assignment_store import ExperimentAssignmentStore

from openhands.core.config.openhands_config import OpenHandsConfig
from openhands.core.logger import openhands_logger as logger


def _get_system_prompt_variant(user_id, conversation_id):
    """Get the system prompt variant for the experiment.

    Args:
        user_id: The user ID
        conversation_id: The conversation ID

    Returns:
        str or None: The PostHog variant name or None if experiment is not enabled or error occurs
    """
    # No-op if the specific experiment is not enabled
    if not EXPERIMENT_SYSTEM_PROMPT_EXPERIMENT:
        logger.info(
            'experiment_manager_002:ab_testing:skipped',
            extra={
                'convo_id': conversation_id,
                'reason': 'experiment_not_enabled',
                'experiment': EXPERIMENT_SYSTEM_PROMPT_EXPERIMENT,
            },
        )
        return None

    # Use experiment name as the flag key
    try:
        enabled_variant = posthog.get_feature_flag(
            EXPERIMENT_SYSTEM_PROMPT_EXPERIMENT, conversation_id
        )
    except Exception as e:
        logger.error(
            'experiment_manager:get_feature_flag:failed',
            extra={
                'convo_id': conversation_id,
                'experiment': EXPERIMENT_SYSTEM_PROMPT_EXPERIMENT,
                'error': str(e),
            },
        )
        return None

    # Store the experiment assignment in the database
    try:
        experiment_store = ExperimentAssignmentStore()
        experiment_store.update_experiment_variant(
            conversation_id=conversation_id,
            experiment_name='system_prompt_experiment',
            variant=enabled_variant,
        )
    except Exception as e:
        logger.error(
            'experiment_manager:store_assignment:failed',
            extra={
                'convo_id': conversation_id,
                'experiment': EXPERIMENT_SYSTEM_PROMPT_EXPERIMENT,
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
            event='system_prompt_set',
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
                'experiment': EXPERIMENT_SYSTEM_PROMPT_EXPERIMENT,
                'error': str(e),
            },
        )
        # Continue execution as this is not critical

    logger.info(
        'posthog_capture',
        extra={
            'event': 'system_prompt_set',
            'posthog_user_id': posthog_user_id,
            'is_feature_env': IS_FEATURE_ENV,
            'conversation_id': conversation_id,
            'variant': enabled_variant,
        },
    )

    return enabled_variant


def handle_system_prompt_experiment(
    user_id, conversation_id, config: OpenHandsConfig
) -> OpenHandsConfig:
    """Handle the system prompt experiment for OpenHands config.

    Args:
        user_id: The user ID
        conversation_id: The conversation ID
        config: The OpenHands configuration

    Returns:
        Modified OpenHands configuration
    """
    enabled_variant = _get_system_prompt_variant(user_id, conversation_id)

    # If variant is None, experiment is not enabled or there was an error
    if enabled_variant is None:
        return config

    # Deep copy the config to avoid modifying the original
    modified_config = copy.deepcopy(config)

    # Set the system prompt filename based on the variant
    if enabled_variant == 'control':
        # Use the long-horizon system prompt for the control variant
        agent_config = modified_config.get_agent_config(modified_config.default_agent)
        agent_config.system_prompt_filename = 'system_prompt_long_horizon.j2'
        agent_config.enable_plan_mode = True
    elif enabled_variant == 'interactive':
        modified_config.get_agent_config(
            modified_config.default_agent
        ).system_prompt_filename = 'system_prompt_interactive.j2'
    elif enabled_variant == 'no_tools':
        modified_config.get_agent_config(
            modified_config.default_agent
        ).system_prompt_filename = 'system_prompt.j2'
    else:
        logger.error(
            'system_prompt_experiment:unknown_variant',
            extra={
                'user_id': user_id,
                'convo_id': conversation_id,
                'variant': enabled_variant,
                'reason': 'no explicit mapping; returning original config',
            },
        )
        return config

    # Log which prompt is being used
    logger.info(
        'system_prompt_experiment:prompt_selected',
        extra={
            'user_id': user_id,
            'convo_id': conversation_id,
            'system_prompt_filename': modified_config.get_agent_config(
                modified_config.default_agent
            ).system_prompt_filename,
            'variant': enabled_variant,
        },
    )

    return modified_config
