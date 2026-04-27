from openhands.storage.files import FileStore
from openhands.storage.google_cloud import GoogleCloudFileStore
from openhands.storage.local import LocalFileStore
from openhands.storage.memory import InMemoryFileStore
from openhands.storage.s3 import S3FileStore


def get_file_store(
    file_store_type: str,
    file_store_path: str | None = None,
) -> FileStore:
    if file_store_type == 'local':
        if file_store_path is None:
            raise ValueError('file_store_path is required for local file store')
        return LocalFileStore(file_store_path)
    elif file_store_type == 's3':
        return S3FileStore(file_store_path)
    elif file_store_type == 'google_cloud':
        return GoogleCloudFileStore(file_store_path)
    else:
        return InMemoryFileStore()
