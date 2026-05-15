# OpenHands FileStore Module

The file store module provides different storage backends for file operations in OpenHands. This module implements a common interface (`FileStore`) that allows for interchangeable storage backends.

All FileStore implementations use `DiscriminatedUnionMixin` for automatic serialization/deserialization with a `kind` discriminator field based on the class name.

## Usage

```python
from openhands.app_server.file_store import get_file_store, LocalFileStore

# Using the factory function
store = get_file_store("local", "/tmp/file_store")

# Or instantiate directly
store = LocalFileStore(root="/tmp/file_store")

# Write, read, list, and delete operations
store.write("example.txt", "Hello, world!")
content = store.read("example.txt")
files = store.list("/")
store.delete("example.txt")
```

## Available Storage Backends

### 1. Local File Storage (`LocalFileStore`)

Local file storage saves files to the local filesystem.

**Parameters:**
- `root`: The root directory for file storage (supports `~` expansion)

### 2. In-Memory Storage (`InMemoryFileStore`)

In-memory storage keeps files in memory, useful for testing or temporary storage.

**Parameters:**
- `files`: Optional dictionary of initial files (default: empty)

### 3. Amazon S3 Storage (`S3FileStore`)

S3 storage uses Amazon S3 or compatible services for file storage.

**Parameters:**
- `bucket`: The S3 bucket name (falls back to `AWS_S3_BUCKET` environment variable if empty)

**Environment Variables:**
- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret key
- `AWS_S3_BUCKET`: Default bucket name (used if `bucket` parameter is empty)
- `AWS_S3_ENDPOINT`: Optional custom endpoint for S3-compatible services
- `AWS_S3_SECURE`: Whether to use HTTPS (default: "true")

### 4. Google Cloud Storage (`GoogleCloudFileStore`)

Google Cloud Storage uses GCS buckets for file storage.

**Parameters:**
- `bucket_name`: The GCS bucket name (falls back to `GOOGLE_CLOUD_BUCKET_NAME` environment variable if empty)

**Environment Variables:**
- `GOOGLE_CLOUD_BUCKET_NAME`: Default bucket name (used if `bucket_name` parameter is empty)
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to Google Cloud credentials JSON file

## Configuration

To configure the file store in OpenHands, use the following configuration options:

```toml
[core]
# File store type: "local", "memory", "s3", "google_cloud"
file_store = "local"

# Path/bucket for file store (interpretation depends on file_store type)
file_store_path = "/tmp/file_store"
```
