# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
import boto3

from openhands.core.logger import openhands_logger as logger


def _create_bedrock_client(
    aws_region_name: str | None = None,
) -> boto3.client:
    """Create a Bedrock client using boto3's default credential chain.

    Credentials are resolved entirely by boto3's default chain, which
    supports (in order): environment variables (AWS_ACCESS_KEY_ID, etc.),
    AWS_PROFILE, assume role / web identity, SSO, shared credentials file,
    container credentials (ECS/EKS), EC2 instance metadata (IAM roles),
    and Bedrock API keys (AWS_BEARER_TOKEN_BEDROCK via bearer token auth).

    Explicit ak/sk from config.toml are already exported to environment
    variables by LLMConfig.model_post_init(), so they are picked up
    automatically — no need to pass them here.
    """
    kwargs: dict = {'service_name': 'bedrock'}
    if aws_region_name:
        kwargs['region_name'] = aws_region_name
    return boto3.client(**kwargs)


def list_foundation_models(
    aws_region_name: str | None = None,
) -> list[str]:
    try:
        client = _create_bedrock_client(aws_region_name)
        foundation_models_list = client.list_foundation_models(
            byOutputModality='TEXT', byInferenceType='ON_DEMAND'
        )
        model_summaries = foundation_models_list['modelSummaries']
        model_ids = set('bedrock/' + model['modelId'] for model in model_summaries)

        # Also list cross-region inference profiles (optional — requires
        # bedrock:ListInferenceProfiles permission, which may not be granted).
        try:
            paginator = client.get_paginator('list_inference_profiles')
            for page in paginator.paginate(typeEquals='SYSTEM_DEFINED'):
                for profile in page.get('inferenceProfileSummaries', []):
                    profile_id = profile.get('inferenceProfileId', '')
                    if profile_id:
                        model_ids.add('bedrock/' + profile_id)
        except Exception as profile_err:
            logger.warning(
                'Could not list Bedrock inference profiles (this is optional): %s',
                profile_err,
            )

        return sorted(model_ids)
    except Exception as err:
        logger.warning(
            '%s. To list Bedrock models, configure AWS credentials via one of: '
            'config.toml [llm] aws_access_key_id/aws_secret_access_key, '
            'environment variables (AWS_ACCESS_KEY_ID, AWS_PROFILE, AWS_ROLE_ARN), '
            'Bedrock API key (AWS_BEARER_TOKEN_BEDROCK), '
            'IAM instance role, or ~/.aws/credentials.',
            err,
        )
        return []


def remove_error_modelId(model_list: list[str]) -> list[str]:
    return list(filter(lambda m: not m.startswith('bedrock'), model_list))
