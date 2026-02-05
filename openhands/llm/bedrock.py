# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import boto3

from openhands.core.logger import openhands_logger as logger


def list_foundation_models(
    aws_region_name: str, aws_access_key_id: str, aws_secret_access_key: str
) -> list[str]:
    try:
        # The AWS bedrock model id is not queried, if no AWS parameters are configured.
        client = boto3.client(
            service_name='bedrock',
            region_name=aws_region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        foundation_models_list = client.list_foundation_models(
            byOutputModality='TEXT', byInferenceType='ON_DEMAND'
        )
        model_summaries = foundation_models_list['modelSummaries']
        return ['bedrock/' + model['modelId'] for model in model_summaries]
    except Exception as err:
        logger.warning(
            '%s. Please config AWS_REGION_NAME AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY'
            ' if you want use bedrock model.',
            err,
        )
        return []


def remove_error_modelId(model_list: list[str]) -> list[str]:
    return list(filter(lambda m: not m.startswith('bedrock'), model_list))
