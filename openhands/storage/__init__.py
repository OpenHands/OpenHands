import os

import httpx

from openhands.storage.batched_web_hook import BatchedWebHookFileStore
from openhands.storage.files import FileStore
from openhands.storage.google_cloud import GoogleCloudFileStore
from openhands.storage.local import LocalFileStore
from openhands.storage.memory import InMemoryFileStore
from openhands.storage.s3 import S3FileStore
from openhands.storage.web_hook import WebHookFileStore
from openhands.utils.http_session import httpx_verify_option


def get_file_store(
    file_store_type: str,
    file_store_path: str | None = None,
    file_store_web_hook_url: str | None = None,
    file_store_web_hook_headers: dict | None = None,
    file_store_web_hook_batch: bool = False,
) -> FileStore:
    store: FileStore
    if file_store_type == 'local':
        if file_store_path is None:
            raise ValueError('file_store_path is required for local file store')
        store = LocalFileStore(file_store_path)
    elif file_store_type == 's3':
        store = S3FileStore(file_store_path)
    elif file_store_type == 'google_cloud':
        store = GoogleCloudFileStore(file_store_path)
    else:
        store = InMemoryFileStore()
    if file_store_web_hook_url:
        if file_store_web_hook_headers is None:
            # Fallback to default headers. Use the session api key if it is defined in the env.
            file_store_web_hook_headers = {}
            if os.getenv('SESSION_API_KEY'):
                file_store_web_hook_headers['X-Session-API-Key'] = os.getenv(
                    'SESSION_API_KEY'
                )

        client = httpx.Client(
            headers=file_store_web_hook_headers or {},
            verify=httpx_verify_option(),
        )

        if file_store_web_hook_batch:
            # Use batched webhook file store
            store = BatchedWebHookFileStore(
                store,
                file_store_web_hook_url,
                client,
            )
        else:
            # Use regular webhook file store
            store = WebHookFileStore(
                store,
                file_store_web_hook_url,
                client,
            )
    return store
