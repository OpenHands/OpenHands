import importlib

import litellm
from pydantic import SecretStr

from openhands.sdk.llm.utils.verified_models import VERIFIED_MODELS
from openhands.sdk.logger import get_logger


def _get_boto3():
    """Get boto3 module if available, otherwise return None."""
    try:
        return importlib.import_module("boto3")
    except ModuleNotFoundError:
        return None


logger = get_logger(__name__)


def _list_bedrock_foundation_models(
    aws_region_name: str, aws_access_key_id: str, aws_secret_access_key: str
) -> list[str]:
    boto3 = _get_boto3()
    if boto3 is None:
        logger.warning(
            "boto3 is not installed. To use Bedrock models,"
            "install with: openhands-sdk[boto3]"
        )
        return []

    try:
        # The AWS bedrock model id is not queried, if no AWS parameters are configured.
        client = boto3.client(
            service_name="bedrock",
            region_name=aws_region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        foundation_models_list = client.list_foundation_models(
            byOutputModality="TEXT", byInferenceType="ON_DEMAND"
        )
        model_summaries = foundation_models_list["modelSummaries"]
        return ["bedrock/" + model["modelId"] for model in model_summaries]
    except Exception as err:
        logger.warning(
            "%s. Please config AWS_REGION_NAME AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY"
            " if you want use bedrock model.",
            err,
        )
        return []


def get_supported_llm_models(
    aws_region_name: str | None = None,
    aws_access_key_id: SecretStr | None = None,
    aws_secret_access_key: SecretStr | None = None,
) -> list[str]:
    """Get all models supported by LiteLLM.

    This function combines models from litellm and Bedrock, removing any
    error-prone Bedrock models.

    Returns:
        list[str]: A sorted list of unique model names.
    """
    litellm_model_list = litellm.model_list + list(litellm.model_cost.keys())
    litellm_model_list_without_bedrock = list(
        filter(lambda m: not m.startswith("bedrock"), litellm_model_list)
    )
    bedrock_model_list = []
    if aws_region_name and aws_access_key_id and aws_secret_access_key:
        bedrock_model_list = _list_bedrock_foundation_models(
            aws_region_name,
            aws_access_key_id.get_secret_value(),
            aws_secret_access_key.get_secret_value(),
        )
    model_list = litellm_model_list_without_bedrock + bedrock_model_list
    return model_list


def _split_is_actually_version(split: list[str]) -> bool:
    return (
        len(split) > 1
        and bool(split[1])
        and bool(split[1][0])
        and split[1][0].isdigit()
    )


def _extract_model_and_provider(model: str) -> tuple[str, str, str]:
    """
    Extract provider and model information from a model identifier.
    """
    separator = "/"
    split = model.split(separator)

    if len(split) == 1:
        # no "/" separator found, try with "."
        separator = "."
        split = model.split(separator)
        if _split_is_actually_version(split):
            split = [separator.join(split)]  # undo the split

    if len(split) == 1:
        matched_provider = ""
        for provider, models in VERIFIED_MODELS.items():
            if split[0] in models:
                matched_provider = provider
                break

        if matched_provider:
            return matched_provider, split[0], "/"

        return matched_provider, model, ""

    provider = split[0]
    model_id = separator.join(split[1:])
    return provider, model_id, separator


def get_unverified_models(
    aws_region_name: str | None = None,
    aws_access_key_id: SecretStr | None = None,
    aws_secret_access_key: SecretStr | None = None,
) -> dict[str, list[str]]:
    """
    Organize a mapping of unverified model identifiers by provider.
    """
    result_dict: dict[str, list[str]] = {}

    models = get_supported_llm_models(
        aws_region_name, aws_access_key_id, aws_secret_access_key
    )
    for model in models:
        provider, model_id, separator = _extract_model_and_provider(model)

        # Ignore "anthropic" providers with a separator of "."
        # These are outdated and incompatible providers.
        if provider == "anthropic" and separator == ".":
            continue

        # Dedup verified models
        if provider in VERIFIED_MODELS and model_id in VERIFIED_MODELS[provider]:
            continue

        key = provider or "other"
        if key not in result_dict:
            result_dict[key] = []

        result_dict[key].append(model_id)

    return result_dict


UNVERIFIED_MODELS_EXCLUDING_BEDROCK = get_unverified_models()
