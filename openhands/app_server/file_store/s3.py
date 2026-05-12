import os
from typing import Any, TypedDict

import boto3
import botocore
from pydantic import Field, PrivateAttr

from openhands.app_server.file_store.files import FileStore


class S3ObjectDict(TypedDict):
    Key: str


class GetObjectOutputDict(TypedDict):
    Body: Any


class ListObjectsV2OutputDict(TypedDict):
    Contents: list[S3ObjectDict] | None


class S3FileStore(FileStore):
    """S3-compatible file store.

    The S3 client is initialized lazily on first access.
    """

    bucket_name: str = Field(default='')

    _client: Any = PrivateAttr(default=None)
    _resolved_bucket: str | None = PrivateAttr(default=None)

    def _get_bucket_name(self) -> str:
        """Get bucket name, falling back to environment variable if not set."""
        if self._resolved_bucket is None:
            self._resolved_bucket = self.bucket_name or os.environ['AWS_S3_BUCKET']
        return self._resolved_bucket

    @property
    def client(self) -> Any:
        """Get the S3 client, initializing lazily on first access."""
        if self._client is None:
            access_key = os.getenv('AWS_ACCESS_KEY_ID')
            secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            secure = os.getenv('AWS_S3_SECURE', 'true').lower() == 'true'
            endpoint = self._ensure_url_scheme(secure, os.getenv('AWS_S3_ENDPOINT'))
            self._client = boto3.client(
                's3',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                endpoint_url=endpoint,
                use_ssl=secure,
            )
        return self._client

    def write(self, path: str, contents: str | bytes) -> None:
        try:
            as_bytes = (
                contents.encode('utf-8') if isinstance(contents, str) else contents
            )
            self.client.put_object(
                Bucket=self._get_bucket_name(), Key=path, Body=as_bytes
            )
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'AccessDenied':
                raise FileNotFoundError(
                    f"Error: Access denied to bucket '{self._get_bucket_name()}'."
                )
            elif e.response['Error']['Code'] == 'NoSuchBucket':
                raise FileNotFoundError(
                    f"Error: The bucket '{self._get_bucket_name()}' does not exist."
                )
            raise FileNotFoundError(
                f"Error: Failed to write to bucket '{self._get_bucket_name()}' at path {path}: {e}"
            )

    def read(self, path: str) -> str:
        try:
            response: GetObjectOutputDict = self.client.get_object(
                Bucket=self._get_bucket_name(), Key=path
            )
            with response['Body'] as stream:
                return str(stream.read().decode('utf-8'))
        except botocore.exceptions.ClientError as e:
            # Catch all S3-related errors
            if e.response['Error']['Code'] == 'NoSuchBucket':
                raise FileNotFoundError(
                    f"Error: The bucket '{self._get_bucket_name()}' does not exist."
                )
            elif e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(
                    f"Error: The object key '{path}' does not exist in bucket '{self._get_bucket_name()}'."
                )
            else:
                raise FileNotFoundError(
                    f"Error: Failed to read from bucket '{self._get_bucket_name()}' at path {path}: {e}"
                )
        except Exception as e:
            raise FileNotFoundError(
                f"Error: Failed to read from bucket '{self._get_bucket_name()}' at path {path}: {e}"
            )

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
        results: set[str] = set()
        prefix_len = len(path)
        response: ListObjectsV2OutputDict = self.client.list_objects_v2(
            Bucket=self._get_bucket_name(), Prefix=path
        )
        contents = response.get('Contents')
        if not contents:
            return []
        paths = [obj['Key'] for obj in contents]
        for sub_path in paths:
            if sub_path == path:
                continue
            try:
                index = sub_path.index('/', prefix_len + 1)
                if index != prefix_len:
                    results.add(sub_path[: index + 1])
            except ValueError:
                results.add(sub_path)
        return list(results)

    def delete(self, path: str) -> None:
        try:
            # Sanitize path
            if not path or path == '/':
                path = ''
            if path.endswith('/'):
                path = path[:-1]

            # Try to delete any child resources (Assume the path is a directory)
            response = self.client.list_objects_v2(
                Bucket=self._get_bucket_name(), Prefix=f'{path}/'
            )
            for content in response.get('Contents') or []:
                self.client.delete_object(
                    Bucket=self._get_bucket_name(), Key=content['Key']
                )

            # Next try to delete item as a file
            self.client.delete_object(Bucket=self._get_bucket_name(), Key=path)

        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchBucket':
                raise FileNotFoundError(
                    f"Error: The bucket '{self._get_bucket_name()}' does not exist."
                )
            elif e.response['Error']['Code'] == 'AccessDenied':
                raise FileNotFoundError(
                    f"Error: Access denied to bucket '{self._get_bucket_name()}'."
                )
            elif e.response['Error']['Code'] == 'NoSuchKey':
                raise FileNotFoundError(
                    f"Error: The object key '{path}' does not exist in bucket '{self._get_bucket_name()}'."
                )
            else:
                raise FileNotFoundError(
                    f"Error: Failed to delete key '{path}' from bucket '{self._get_bucket_name()}': {e}"
                )
        except Exception as e:
            raise FileNotFoundError(
                f"Error: Failed to delete key '{path}' from bucket '{self._get_bucket_name()}: {e}"
            )

    def _ensure_url_scheme(self, secure: bool, url: str | None) -> str | None:
        if not url:
            return None
        if secure:
            if not url.startswith('https://'):
                url = 'https://' + url.removeprefix('http://')
        else:
            if not url.startswith('http://'):
                url = 'http://' + url.removeprefix('https://')
        return url
