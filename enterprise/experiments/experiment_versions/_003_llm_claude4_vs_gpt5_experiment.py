"""LiteLLM model experiment handler.

This module contains the handler for the LiteLLM model experiment.
"""

import posthog
from experiments.constants import EXPERIMENT_CLAUDE4_VS_GPT5
from server.constants import (
    IS_FEATURE_ENV,
    build_litellm_proxy_model_path,
    get_default_litellm_model,
)
from storage.experiment_assignment_store import ExperimentAssignmentStore

from openhands.core.logger import openhands_logger as logger
from openhands.server.session.conversation_init_data import ConversationInitData


def _get_model_variant(user_id: str | None, conversation_id: str) -> str | None:
    if not EXPERIMENT_CLAUDE4_VS_GPT5:
        logger.info(
            'experiment_manager:ab_testing:skipped',
            extra={
                'convo_id': conversation_id,
                'reason': 'experiment_not_enabled',
                'experiment': EXPERIMENT_CLAUDE4_VS_GPT5,
            },
        )
        return None

    try:
        enabled_variant = posthog.get_feature_flag(
            EXPERIMENT_CLAUDE4_VS_GPT5, conversation_id
        )
    except Exception as e:
        logger.error(
            'experiment_manager:get_feature_flag:failed',
            extra={
                'convo_id': conversation_id,
                'experiment': EXPERIMENT_CLAUDE4_VS_GPT5,
                'error': str(e),
            },
        )
        return None

    # Store the experiment assignment in the database
    try:
        experiment_store = ExperimentAssignmentStore()
        experiment_store.update_experiment_variant(
            conversation_id=conversation_id,
            experiment_name='claude4_vs_gpt5_experiment',
            variant=enabled_variant,
        )
    except Exception as e:
        logger.error(
            'experiment_manager:store_assignment:failed',
            extra={
                'convo_id': conversation_id,
                'experiment': EXPERIMENT_CLAUDE4_VS_GPT5,
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
            event='claude4_or_gpt5_set',
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
                'experiment': EXPERIMENT_CLAUDE4_VS_GPT5,
                'error': str(e),
            },
        )
        # Continue execution as this is not critical

    logger.info(
        'posthog_capture',
        extra={
            'event': 'claude4_or_gpt5_set',
            'posthog_user_id': posthog_user_id,
            'is_feature_env': IS_FEATURE_ENV,
            'conversation_id': conversation_id,
            'variant': enabled_variant,
        },
    )

    return enabled_variant


def handle_claude4_vs_gpt5_experiment(
    user_id: str | None,
    conversation_id: str,
    conversation_settings: ConversationInitData,
) -> ConversationInitData:
    """Handle the LiteLLM model experiment.

    Args:
        user_id: The user ID
        conversation_id: The conversation ID
        conversation_settings: The conversation settings

    Returns:
        Modified conversation settings
    """
    enabled_variant = _get_model_variant(user_id, conversation_id)

    if not enabled_variant:
        return conversation_settings

    # Set the model based on the feature flag variant
    if enabled_variant == 'gpt5':
        model = build_litellm_proxy_model_path('gpt-5-2025-08-07')
        conversation_settings.llm_model = model
    else:
        conversation_settings.llm_model = get_default_litellm_model()

    return conversation_settings
