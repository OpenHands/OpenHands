from __future__ import annotations

import logging
import shutil
import tempfile
import threading
from abc import ABC
from dataclasses import dataclass, field
from io import BytesIO, StringIO
from unittest import TestCase
from unittest.mock import patch

import botocore.exceptions
from google.api_core.exceptions import NotFound

from openhands.storage.files import FileStore
from openhands.storage.google_cloud import GoogleCloudFileStore
from openhands.storage.local import LocalFileStore
from openhands.storage.memory import InMemoryFileStore
from openhands.storage.s3 import S3FileStore


class _StorageTest(ABC):
    store: FileStore

    def get_store(self) -> FileStore:
        try:
            self.store.delete('')
        except Exception:
            pass
        return self.store

    def test_basic_fileops(self):
        filename = 'test.txt'
        store = self.get_store()
        store.write(filename, 'Hello, world!')
        self.assertEqual(store.read(filename), 'Hello, world!')
        self.assertEqual(store.list(''), [filename])
        store.delete(filename)
        with self.assertRaises(FileNotFoundError):
            store.read(filename)

    def test_complex_path_fileops(self):
        filenames = ['foo.bar.baz', './foo/bar/baz', 'foo/bar/baz', '/foo/bar/baz']
        store = self.get_store()
        for filename in filenames:
            store.write(filename, 'Hello, world!')
            self.assertEqual(store.read(filename), 'Hello, world!')
            store.delete(filename)
            with self.assertRaises(FileNotFoundError):
                store.read(filename)

    def test_list(self):
        store = self.get_store()
        store.write('foo.txt', 'Hello, world!')
        store.write('bar.txt', 'Hello, world!')
        store.write('baz.txt', 'Hello, world!')
        file_names = store.list('')
        file_names.sort()
        self.assertEqual(file_names, ['bar.txt', 'baz.txt', 'foo.txt'])
        store.delete('foo.txt')
        store.delete('bar.txt')
        store.delete('baz.txt')

    def test_deep_list(self):
        store = self.get_store()
        store.write('foo/bar/baz.txt', 'Hello, world!')
        store.write('foo/bar/qux.txt', 'Hello, world!')
        store.write('foo/bar/quux.txt', 'Hello, world!')
        self.assertEqual(store.list(''), ['foo/'])
        self.assertEqual(store.list('foo'), ['foo/bar/'])
        file_names = store.list('foo/bar')
        file_names.sort()
        self.assertEqual(
            file_names, ['foo/bar/baz.txt', 'foo/bar/quux.txt', 'foo/bar/qux.txt']
        )
        store.delete('foo/bar/baz.txt')
        store.delete('foo/bar/qux.txt')
        store.delete('foo/bar/quux.txt')

    def test_directory_deletion(self):
        store = self.get_store()
        # Create a directory structure
        store.write('foo/bar/baz.txt', 'Hello, world!')
        store.write('foo/bar/qux.txt', 'Hello, world!')
        store.write('foo/other.txt', 'Hello, world!')
        store.write('foo/bar/subdir/file.txt', 'Hello, world!')

        # Verify initial structure
        self.assertEqual(store.list(''), ['foo/'])
        self.assertEqual(sorted(store.list('foo')), ['foo/bar/', 'foo/other.txt'])
        self.assertEqual(
            sorted(store.list('foo/bar')),
            ['foo/bar/baz.txt', 'foo/bar/qux.txt', 'foo/bar/subdir/'],
        )

        # Delete a directory
        store.delete('foo/bar')

        # Verify directory and its contents are gone, but other files remain
        self.assertEqual(store.list(''), ['foo/'])
        self.assertEqual(store.list('foo'), ['foo/other.txt'])

        # Delete root directory
        store.delete('foo')

        # Verify everything is gone
        self.assertEqual(store.list(''), [])


class TestLocalFileStore(TestCase, _StorageTest):
    def setUp(self):
        # Create a unique temporary directory for each test instance
        self.temp_dir = tempfile.mkdtemp(prefix='openhands_test_')
        self.store = LocalFileStore(self.temp_dir)

    def tearDown(self):
        try:
            # Use ignore_errors=True to avoid failures if directory is not empty
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            logging.warning(
                f'Failed to remove temporary directory {self.temp_dir}: {e}'
            )

    def test_concurrent_writes_no_corruption(self):
        """Test that concurrent writes don't corrupt file content.

        This test verifies the atomic write fix by having 9 threads write
        progressively shorter strings to the same file simultaneously.
        Without atomic writes, a shorter write following a longer write
        could result in corrupted content (e.g., "123" followed by garbage
        from the previous longer write).

        The final content must be exactly one of the valid strings written,
        with no trailing garbage from other writes.
        """
        filename = 'concurrent_test.txt'
        # Strings from longest to shortest: "123456789", "12345678", ..., "1"
        valid_contents = ['123456789'[:i] for i in range(9, 0, -1)]
        errors: list[Exception] = []
        barrier = threading.Barrier(len(valid_contents))

        def write_content(content: str):
            try:
                # Wait for all threads to be ready before writing
                barrier.wait()
                self.store.write(filename, content)
            except Exception as e:
                errors.append(e)

        # Start all threads
        threads = [
            threading.Thread(target=write_content, args=(content,))
            for content in valid_contents
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Check for errors during writes
        self.assertEqual(
            errors, [], f'Errors occurred during concurrent writes: {errors}'
        )

        # Read final content and verify it's one of the valid strings
        final_content = self.store.read(filename)
        self.assertIn(
            final_content,
            valid_contents,
            f"File content '{final_content}' is not one of the valid strings. "
            f'Length: {len(final_content)}. This indicates file corruption from '
            f'concurrent writes (e.g., shorter write did not fully replace longer write).',
        )


class TestInMemoryFileStore(TestCase, _StorageTest):
    def setUp(self):
        self.store = InMemoryFileStore()


class TestGoogleCloudFileStore(TestCase, _StorageTest):
    def setUp(self):
        with patch('google.cloud.storage.Client', _MockGoogleCloudClient):
            self.store = GoogleCloudFileStore('dear-liza')


class TestS3FileStore(TestCase, _StorageTest):
    def setUp(self):
        with patch('boto3.client', lambda service, **kwargs: _MockS3Client()):
            self.store = S3FileStore('dear-liza')


# I would have liked to use cloud-storage-mocker here but the python versions were incompatible :(
# If we write tests for the S3 storage class I would definitely recommend we use moto.
class _MockGoogleCloudClient:
    def bucket(self, name: str):
        assert name == 'dear-liza'
        return _MockGoogleCloudBucket()


@dataclass
class _MockGoogleCloudBucket:
    blobs_by_path: dict[str, _MockGoogleCloudBlob] = field(default_factory=dict)

    def blob(self, path: str | None = None) -> _MockGoogleCloudBlob:
        return self.blobs_by_path.get(path) or _MockGoogleCloudBlob(self, path)

    def list_blobs(self, prefix: str | None = None) -> list[_MockGoogleCloudBlob]:
        blobs = list(self.blobs_by_path.values())
        if prefix and prefix != '/':
            blobs = [blob for blob in blobs if blob.name.startswith(prefix)]
        return blobs


@dataclass
class _MockGoogleCloudBlob:
    bucket: _MockGoogleCloudBucket
    name: str
    content: str | bytes | None = None

    def open(self, op: str):
        if op == 'r':
            if self.content is None:
                raise FileNotFoundError()
            return StringIO(self.content)
        if op == 'w':
            return _MockGoogleCloudBlobWriter(self)

    def delete(self):
        if self.name not in self.bucket.blobs_by_path:
            raise NotFound('Blob not found')
        del self.bucket.blobs_by_path[self.name]


@dataclass
class _MockGoogleCloudBlobWriter:
    blob: _MockGoogleCloudBlob
    content: str | bytes = None

    def __enter__(self):
        return self

    def write(self, __b):
        assert (
            self.content is None
        )  # We don't support buffered writes in this mock for now, as it is not needed
        self.content = __b

    def __exit__(self, exc_type, exc_val, exc_tb):
        blob = self.blob
        blob.content = self.content
        blob.bucket.blobs_by_path[blob.name] = blob


class _MockS3Client:
    def __init__(self):
        self.objects_by_bucket: dict[str, dict[str, _MockS3Object]] = {}

    def put_object(self, Bucket: str, Key: str, Body: str | bytes) -> None:
        if Bucket not in self.objects_by_bucket:
            self.objects_by_bucket[Bucket] = {}
        self.objects_by_bucket[Bucket][Key] = _MockS3Object(Key, Body)

    def get_object(self, Bucket: str, Key: str) -> dict:
        if Bucket not in self.objects_by_bucket:
            raise botocore.exceptions.ClientError(
                {
                    'Error': {
                        'Code': 'NoSuchBucket',
                        'Message': f"The bucket '{Bucket}' does not exist",
                    }
                },
                'GetObject',
            )
        if Key not in self.objects_by_bucket[Bucket]:
            raise botocore.exceptions.ClientError(
                {
                    'Error': {
                        'Code': 'NoSuchKey',
                        'Message': f"The specified key '{Key}' does not exist",
                    }
                },
                'GetObject',
            )
        content = self.objects_by_bucket[Bucket][Key].content
        if isinstance(content, bytes):
            return {'Body': BytesIO(content)}
        return {'Body': StringIO(content)}

    def list_objects_v2(self, Bucket: str, Prefix: str = '') -> dict:
        if Bucket not in self.objects_by_bucket:
            raise botocore.exceptions.ClientError(
                {
                    'Error': {
                        'Code': 'NoSuchBucket',
                        'Message': f"The bucket '{Bucket}' does not exist",
                    }
                },
                'ListObjectsV2',
            )
        objects = self.objects_by_bucket[Bucket]
        contents = [
            {'Key': key}
            for key in objects.keys()
            if not Prefix or key.startswith(Prefix)
        ]
        return {'Contents': contents} if contents else {}

    def delete_object(self, Bucket: str, Key: str) -> None:
        if Bucket not in self.objects_by_bucket:
            raise botocore.exceptions.ClientError(
                {
                    'Error': {
                        'Code': 'NoSuchBucket',
                        'Message': f"The bucket '{Bucket}' does not exist",
                    }
                },
                'DeleteObject',
            )
        self.objects_by_bucket[Bucket].pop(Key, None)


@dataclass
class _MockS3Object:
    key: str
    content: str | bytes
