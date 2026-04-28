import os

from google.api_core.exceptions import NotFound
from google.cloud import storage
from google.cloud.storage.blob import Blob
from google.cloud.storage.bucket import Bucket
from google.cloud.storage.client import Client
from pydantic import Field, PrivateAttr

from openhands.app_server.file_store.files import FileStore


class GoogleCloudFileStore(FileStore):
    """Google Cloud Storage file store.

    If GOOGLE_APPLICATION_CREDENTIALS is defined in the environment it will be used
    for authentication. Otherwise access will be anonymous.

    The storage client and bucket are initialized lazily on first access.
    """

    bucket_name: str = Field(default='')

    _storage_client: Client | None = PrivateAttr(default=None)
    _bucket: Bucket | None = PrivateAttr(default=None)

    def _get_bucket_name(self) -> str:
        """Get bucket name, falling back to environment variable if not set."""
        if self.bucket_name:
            return self.bucket_name
        return os.environ['GOOGLE_CLOUD_BUCKET_NAME']

    @property
    def storage_client(self) -> Client:
        """Get the storage client, initializing lazily on first access."""
        if self._storage_client is None:
            self._storage_client = storage.Client()
        return self._storage_client

    @property
    def bucket(self) -> Bucket:
        """Get the bucket, initializing lazily on first access."""
        if self._bucket is None:
            self._bucket = self.storage_client.bucket(self._get_bucket_name())
        return self._bucket

    def write(self, path: str, contents: str | bytes) -> None:
        blob: Blob = self.bucket.blob(path)
        mode = 'wb' if isinstance(contents, bytes) else 'w'
        with blob.open(mode) as f:
            f.write(contents)

    def read(self, path: str) -> str:
        blob: Blob = self.bucket.blob(path)
        try:
            with blob.open('r') as f:
                return str(f.read())
        except NotFound as err:
            raise FileNotFoundError(err)

    def list(self, path: str) -> list[str]:
        if not path or path == '/':
            path = ''
        elif not path.endswith('/'):
            path += '/'
        # The delimiter logic screens out directories, so we can't use it. :(
        # For example, given a structure:
        #   foo/bar/zap.txt
        #   foo/bar/bang.txt
        #   ping.txt
        # prefix=None, delimiter="/"   yields  ["ping.txt"]  # :(
        # prefix="foo", delimiter="/"  yields  []  # :(
        blobs: set[str] = set()
        prefix_len = len(path)
        for blob in self.bucket.list_blobs(prefix=path):
            name: str = blob.name
            if name == path:
                continue
            try:
                index = name.index('/', prefix_len + 1)
                if index != prefix_len:
                    blobs.add(name[: index + 1])
            except ValueError:
                blobs.add(name)
        return list(blobs)

    def delete(self, path: str) -> None:
        # Sanitize path
        if not path or path == '/':
            path = ''
        if path.endswith('/'):
            path = path[:-1]

        # Try to delete any child resources (Assume the path is a directory)
        for blob in self.bucket.list_blobs(prefix=f'{path}/'):
            blob.delete()

        # Next try to delete item as a file
        try:
            file_blob: Blob = self.bucket.blob(path)
            file_blob.delete()
        except NotFound:
            pass
