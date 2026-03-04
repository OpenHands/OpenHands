"""LiteLLM model experiment handler.

This module contains the handler for the LiteLLM model experiment.
"""

import posthog
from experiments.constants import EXPERIMENT_LITELLM_DEFAULT_MODEL_EXPERIMENT
from server.constants import (
    IS_FEATURE_ENV,
    build_litellm_proxy_model_path,
    get_default_litellm_model,
)

from openhands.core.logger import openhands_logger as logger


def handle_litellm_default_model_experiment(
    user_id, conversation_id, conversation_settings
):
    """Handle the LiteLLM model experiment.

    Args:
        user_id: The user ID
        conversation_id: The conversation ID
        conversation_settings: The conversation settings

    Returns:
        Modified conversation settings
    """
    # No-op if the specific experiment is not enabled
    if not EXPERIMENT_LITELLM_DEFAULT_MODEL_EXPERIMENT:
        logger.info(
            'experiment_manager:ab_testing:skipped',
            extra={
                'convo_id': conversation_id,
                'reason': 'experiment_not_enabled',
                'experiment': EXPERIMENT_LITELLM_DEFAULT_MODEL_EXPERIMENT,
            },
        )
        return conversation_settings

    # Use experiment name as the flag key
    try:
        enabled_variant = posthog.get_feature_flag(
            EXPERIMENT_LITELLM_DEFAULT_MODEL_EXPERIMENT, conversation_id
        )
    except Exception as e:
        logger.error(
            'experiment_manager:get_feature_flag:failed',
            extra={
                'convo_id': conversation_id,
                'experiment': EXPERIMENT_LITELLM_DEFAULT_MODEL_EXPERIMENT,
                'error': str(e),
            },
        )
        return conversation_settings

    # Log the experiment event
    # If this is a feature environment, add "FEATURE_" prefix to user_id for PostHog
    posthog_user_id = f'FEATURE_{user_id}' if IS_FEATURE_ENV else user_id

    try:
        posthog.capture(
            distinct_id=posthog_user_id,
            event='model_set',
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
                'experiment': EXPERIMENT_LITELLM_DEFAULT_MODEL_EXPERIMENT,
                'error': str(e),
            },
        )
        # Continue execution as this is not critical

    logger.info(
        'posthog_capture',
        extra={
            'event': 'model_set',
            'posthog_user_id': posthog_user_id,
            'is_feature_env': IS_FEATURE_ENV,
            'conversation_id': conversation_id,
            'variant': enabled_variant,
        },
    )

    # Set the model based on the feature flag variant
    if enabled_variant == 'claude37':
        # Use the shared utility to construct the LiteLLM proxy model path
        model = build_litellm_proxy_model_path('claude-3-7-sonnet-20250219')
        # Update the conversation settings with the selected model
        conversation_settings.llm_model = model
    else:
        # Update the conversation settings with the default model for the current version
        conversation_settings.llm_model = get_default_litellm_model()

    return conversation_settings
