import httpx
import tenacity

from openhands.storage.files import FileStore
from openhands.utils.async_utils import EXECUTOR
from openhands.utils.http_session import httpx_verify_option


class WebHookFileStore(FileStore):
    """File store which includes a web hook to be invoked after any changes occur.

    This class wraps another FileStore implementation and sends HTTP requests
    to a specified URL whenever files are written or deleted.

    Attributes:
        file_store: The underlying FileStore implementation
        base_url: The base URL for webhook requests
        client: The HTTP client used to make webhook requests
    """

    file_store: FileStore
    base_url: str
    client: httpx.Client

    def __init__(
        self, file_store: FileStore, base_url: str, client: httpx.Client | None = None
    ):
        """Initialize a WebHookFileStore.

        Args:
            file_store: The underlying FileStore implementation
            base_url: The base URL for webhook requests
            client: Optional HTTP client to use for requests. If None, a new client will be created.
        """
        self.file_store = file_store
        self.base_url = base_url
        if client is None:
            client = httpx.Client(verify=httpx_verify_option())
        self.client = client

    def write(self, path: str, contents: str | bytes) -> None:
        """Write contents to a file and trigger a webhook.

        Args:
            path: The path to write to
            contents: The contents to write
        """
        self.file_store.write(path, contents)
        EXECUTOR.submit(self._on_write, path, contents)

    def read(self, path: str) -> str:
        """Read contents from a file.

        Args:
            path: The path to read from

        Returns:
            The contents of the file
        """
        return self.file_store.read(path)

    def list(self, path: str) -> list[str]:
        """List files in a directory.

        Args:
            path: The directory path to list

        Returns:
            A list of file paths
        """
        return self.file_store.list(path)

    def delete(self, path: str) -> None:
        """Delete a file and trigger a webhook.

        Args:
            path: The path to delete
        """
        self.file_store.delete(path)
        EXECUTOR.submit(self._on_delete, path)

    @tenacity.retry(
        wait=tenacity.wait_fixed(1),
        stop=tenacity.stop_after_attempt(3),
    )
    def _on_write(self, path: str, contents: str | bytes) -> None:
        """Send a POST request to the webhook URL when a file is written.

        This method is retried up to 3 times with a 1-second delay between attempts.

        Args:
            path: The path that was written to
            contents: The contents that were written

        Raises:
            httpx.HTTPStatusError: If the webhook request fails
        """
        base_url = self.base_url + path
        response = self.client.post(base_url, content=contents)
        response.raise_for_status()

    @tenacity.retry(
        wait=tenacity.wait_fixed(1),
        stop=tenacity.stop_after_attempt(3),
    )
    def _on_delete(self, path: str) -> None:
        """Send a DELETE request to the webhook URL when a file is deleted.

        This method is retried up to 3 times with a 1-second delay between attempts.

        Args:
            path: The path that was deleted

        Raises:
            httpx.HTTPStatusError: If the webhook request fails
        """
        base_url = self.base_url + path
        response = self.client.delete(base_url)
        response.raise_for_status()
