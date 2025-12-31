import time
from functools import lru_cache
from logging import getLogger

import httpx
from litellm.types.utils import ModelInfo
from litellm.utils import get_model_info
from pydantic import SecretStr


logger = getLogger(__name__)


@lru_cache
def _get_model_info_from_litellm_proxy(
    secret_api_key: SecretStr | str | None,
    base_url: str,
    model: str,
    cache_key: int | None = None,
):
    logger.debug(f"Get model_info_from_litellm_proxy:{cache_key}")
    try:
        headers = {}
        if isinstance(secret_api_key, SecretStr):
            secret_api_key = secret_api_key.get_secret_value()
        if secret_api_key:
            headers["Authorization"] = f"Bearer {secret_api_key}"

        response = httpx.get(f"{base_url}/v1/model/info", headers=headers)
        data = response.json().get("data", [])
        current = next(
            (
                info
                for info in data
                if info["model_name"] == model.removeprefix("litellm_proxy/")
            ),
            None,
        )
        if current:
            model_info = current.get("model_info")
            logger.debug(f"Got model info from litellm proxy: {model_info}")
            return model_info
    except Exception as e:
        logger.debug(
            f"Error fetching model info from proxy: {e}",
            exc_info=True,
            stack_info=True,
        )


def get_litellm_model_info(
    secret_api_key: SecretStr | str | None, base_url: str | None, model: str
) -> ModelInfo | None:
    # Try to get model info via openrouter or litellm proxy first
    try:
        if model.startswith("openrouter"):
            model_info = get_model_info(model)
            if model_info:
                return model_info
    except Exception as e:
        logger.debug(f"get_model_info(openrouter) failed: {e}")

    if model.startswith("litellm_proxy/") and base_url:
        # Use the current hour as a cache key - only refresh hourly
        cache_key = int(time.time() / 3600)

        model_info = _get_model_info_from_litellm_proxy(
            secret_api_key=secret_api_key,
            base_url=base_url,
            model=model,
            cache_key=cache_key,
        )
        if model_info:
            return model_info

    # Fallbacks: try base name variants
    try:
        model_info = get_model_info(model.split(":")[0])
        if model_info:
            return model_info
    except Exception:
        pass
    try:
        model_info = get_model_info(model.split("/")[-1])
        if model_info:
            return model_info
    except Exception:
        pass

    return None
