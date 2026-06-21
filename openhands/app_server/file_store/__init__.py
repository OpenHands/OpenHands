from openhands.app_server.file_store.files import FileStore
from openhands.app_server.file_store.google_cloud import GoogleCloudFileStore
from openhands.app_server.file_store.local import LocalFileStore
from openhands.app_server.file_store.memory import InMemoryFileStore
from openhands.app_server.file_store.s3 import S3FileStore


def get_file_store(
    file_store_type: str,
    file_store_path: str | None = None,
) -> FileStore:
    if file_store_type == 'local':
        if file_store_path is None:
            raise ValueError('file_store_path is required for local file store')
        return LocalFileStore(root=file_store_path)
    elif file_store_type == 's3':
        return S3FileStore(bucket_name=file_store_path or '')
    elif file_store_type == 'google_cloud':
        return GoogleCloudFileStore(bucket_name=file_store_path or '')
    else:
        return InMemoryFileStore()
