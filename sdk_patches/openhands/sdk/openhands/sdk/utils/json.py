import json
from datetime import datetime
from typing import Any

from litellm.types.utils import ModelResponse

from openhands.sdk.llm.exceptions import LLMResponseError
from openhands.sdk.llm.utils.metrics import Metrics


class OpenHandsJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles datetime and other OH objects"""

    def default(self, o: object) -> Any:
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, Metrics):
            return o.get()
        if isinstance(o, ModelResponse):
            return o.model_dump()
        return super().default(o)


# Create a single reusable encoder instance
_json_encoder = OpenHandsJSONEncoder()


def dumps(obj, **kwargs):
    """Serialize an object to str format"""
    if not kwargs:
        return _json_encoder.encode(obj)

    # Create a copy of the kwargs to avoid modifying the original
    encoder_kwargs = kwargs.copy()

    # If cls is specified, use it; otherwise use our custom encoder
    if "cls" not in encoder_kwargs:
        encoder_kwargs["cls"] = OpenHandsJSONEncoder

    return json.dumps(obj, **encoder_kwargs)


def loads(json_str, **kwargs):
    """Create a JSON object from str"""
    try:
        return json.loads(json_str, **kwargs)
    except json.JSONDecodeError:
        raise LLMResponseError("No valid JSON object found in response.")
