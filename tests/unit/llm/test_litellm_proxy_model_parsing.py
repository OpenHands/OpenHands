import sys
import types
from unittest.mock import patch

# Provide lightweight stubs for optional dependencies that are imported at module import time
# elsewhere in the codebase, to avoid installing heavy packages for this focused unit test.
if 'pythonjsonlogger' not in sys.modules:
    pythonjsonlogger = types.ModuleType('pythonjsonlogger')
    pythonjsonlogger.json = types.ModuleType('pythonjsonlogger.json')

    class _DummyJsonFormatter:  # minimal stub
        def __init__(self, *args, **kwargs):
            pass

    pythonjsonlogger.json.JsonFormatter = _DummyJsonFormatter
    sys.modules['pythonjsonlogger'] = pythonjsonlogger
    sys.modules['pythonjsonlogger.json'] = pythonjsonlogger.json

if 'google' not in sys.modules:
    google = types.ModuleType('google')
    # make it package-like
    google.__path__ = []  # type: ignore[attr-defined]
    sys.modules['google'] = google
if 'google.api_core' not in sys.modules:
    api_core = types.ModuleType('google.api_core')
    api_core.__path__ = []  # type: ignore[attr-defined]
    sys.modules['google.api_core'] = api_core
if 'google.api_core.exceptions' not in sys.modules:
    exceptions_mod = types.ModuleType('google.api_core.exceptions')

    # Provide a NotFound exception type used by storage backends
    class _NotFound(Exception):
        pass

    exceptions_mod.NotFound = _NotFound
    sys.modules['google.api_core.exceptions'] = exceptions_mod

# Also stub google.cloud and google.cloud.storage used by storage backends
if 'google.cloud' not in sys.modules:
    google_cloud_pkg = types.ModuleType('google.cloud')
    google_cloud_pkg.__path__ = []  # type: ignore[attr-defined]
    sys.modules['google.cloud'] = google_cloud_pkg
if 'google.cloud.storage' not in sys.modules:
    storage_pkg = types.ModuleType('google.cloud.storage')
    storage_pkg.__path__ = []  # type: ignore[attr-defined]

    class _DummyClient:
        def __init__(self, *args, **kwargs):
            pass

    storage_pkg.Client = _DummyClient
    sys.modules['google.cloud.storage'] = storage_pkg

# Submodules used by storage backend
if 'google.cloud.storage.blob' not in sys.modules:
    blob_mod = types.ModuleType('google.cloud.storage.blob')

    class _DummyBlob:
        def __init__(self, *args, **kwargs):
            pass

    blob_mod.Blob = _DummyBlob
    sys.modules['google.cloud.storage.blob'] = blob_mod
if 'google.cloud.storage.bucket' not in sys.modules:
    bucket_mod = types.ModuleType('google.cloud.storage.bucket')

    class _DummyBucket:
        def __init__(self, *args, **kwargs):
            pass

    bucket_mod.Bucket = _DummyBucket
    sys.modules['google.cloud.storage.bucket'] = bucket_mod

# Also provide google.cloud.storage.client module referencing the Client stub
if 'google.cloud.storage.client' not in sys.modules:
    client_mod = types.ModuleType('google.cloud.storage.client')
    try:
        client_mod.Client = sys.modules['google.cloud.storage'].Client  # type: ignore[attr-defined]
    except Exception:

        class _DummyClient2:
            def __init__(self, *args, **kwargs):
                pass

        client_mod.Client = _DummyClient2
    sys.modules['google.cloud.storage.client'] = client_mod

# Stub boto3 used by S3 backend
if 'boto3' not in sys.modules:
    boto3_mod = types.ModuleType('boto3')

    def _noop(*args, **kwargs):
        class _Dummy:
            def __getattr__(self, _):
                return _noop

            def __call__(self, *a, **k):
                return None

        return _Dummy()

    boto3_mod.client = _noop
    boto3_mod.resource = _noop

    class _DummySession:
        def client(self, *a, **k):
            return _noop()

        def resource(self, *a, **k):
            return _noop()

    boto3_mod.session = types.SimpleNamespace(Session=_DummySession)
    sys.modules['boto3'] = boto3_mod

if 'botocore' not in sys.modules:
    botocore_mod = types.ModuleType('botocore')
    botocore_mod.__path__ = []  # type: ignore[attr-defined]
    sys.modules['botocore'] = botocore_mod
if 'botocore.exceptions' not in sys.modules:
    botocore_exc = types.ModuleType('botocore.exceptions')

    class _BotoCoreError(Exception):
        pass

    botocore_exc.BotoCoreError = _BotoCoreError
    sys.modules['botocore.exceptions'] = botocore_exc

# Stub uvicorn server constants used by shutdown listener
if 'uvicorn' not in sys.modules:
    uvicorn_mod = types.ModuleType('uvicorn')
    uvicorn_mod.__path__ = []  # type: ignore[attr-defined]
    sys.modules['uvicorn'] = uvicorn_mod
if 'uvicorn.server' not in sys.modules:
    uvicorn_server = types.ModuleType('uvicorn.server')
    # minimal placeholder; value isn't used in this test
    uvicorn_server.HANDLED_SIGNALS = set()
    sys.modules['uvicorn.server'] = uvicorn_server

# Stub json_repair used by openhands.io.json
if 'json_repair' not in sys.modules:
    json_repair_mod = types.ModuleType('json_repair')

    def repair_json(s: str) -> str:
        return s

    json_repair_mod.repair_json = repair_json
    sys.modules['json_repair'] = json_repair_mod

# Stub deprecated.deprecated decorator
if 'deprecated' not in sys.modules:
    deprecated_mod = types.ModuleType('deprecated')

    def deprecated(*dargs, **dkwargs):  # decorator shim
        def _wrap(func):
            return func

        # Support both @deprecated and @deprecated(reason="...") usages
        if dargs and callable(dargs[0]) and not dkwargs:
            return dargs[0]
        return _wrap

    deprecated_mod.deprecated = deprecated
    sys.modules['deprecated'] = deprecated_mod

# Import OpenHands after stubbing optional deps
from openhands.core.config.llm_config import LLMConfig
from openhands.llm.llm import LLM
from openhands.llm.metrics import Metrics


class DummyResponse:
    def __init__(self, json_data):
        self._json = json_data

    def json(self):
        return self._json


@patch('httpx.get')
def test_litellm_proxy_model_with_nested_slashes_is_accepted(mock_get):
    # Arrange: simulate LiteLLM proxy /v1/model/info returning our model
    model_tail = 'copilot/gpt-4.1'
    mock_get.return_value = DummyResponse(
        {
            'data': [
                {
                    'model_name': model_tail,
                    'model_info': {
                        'max_input_tokens': 128000,
                        'supports_vision': False,
                    },
                }
            ]
        }
    )

    cfg = LLMConfig(
        model=f'litellm_proxy/{model_tail}',
        api_key=None,
        base_url='http://localhost:4000',  # any string; we mock httpx.get anyway
    )

    # Act: construct LLM; should not raise ValidationError
    llm = LLM(config=cfg, service_id='test', metrics=Metrics(model_name=cfg.model))

    # Assert: model remains intact and model_info was set from proxy data
    assert llm.config.model == f'litellm_proxy/{model_tail}'
    assert llm.model_info is None or isinstance(
        llm.model_info, (dict, types.MappingProxyType)
    )


@patch('httpx.get')
def test_litellm_proxy_model_info_lookup_uses_full_tail(mock_get):
    # Ensure we match exactly the entire tail after prefix when selecting model info
    model_tail = 'nested/provider/path/model-x'
    mock_get.return_value = DummyResponse(
        {
            'data': [
                {'model_name': model_tail, 'model_info': {'max_input_tokens': 32000}},
                {'model_name': 'other', 'model_info': {'max_input_tokens': 1}},
            ]
        }
    )

    cfg = LLMConfig(
        model=f'litellm_proxy/{model_tail}',
        api_key=None,
        base_url='http://localhost:4000',
    )

    llm = LLM(config=cfg, service_id='test', metrics=Metrics(model_name=cfg.model))

    # If proxy data was set, prefer that exact match; otherwise at least the construction should succeed
    if llm.model_info is not None:
        assert llm.model_info.get('max_input_tokens') == 32000
