"""TCP port forwarder for Codespace sandbox worker ports.

In GitHub Codespaces, Docker-proxy binds on the VM host network namespace
while the Codespace port-forwarding agent runs inside the devcontainer.
This means the agent never detects sandbox worker ports and Codespace
tunnel URLs don't work.

This module starts asyncio TCP forwarders *inside* the devcontainer that
listen on ``0.0.0.0:<host_port>`` and forward traffic to the sandbox
container via the shared Docker network (``<container_name>:<container_port>``).
The Codespace agent detects these local listeners and auto-forwards them
through the tunnel.
"""

import asyncio
import functools
import logging

_logger = logging.getLogger(__name__)

_PIPE_BUF_SIZE = 65536


class TcpPortForwarderManager:
    """Manages TCP port forwarders for sandbox containers.

    Each sandbox can have multiple forwarded ports.  Call
    :meth:`start_forwarders` after a sandbox starts and
    :meth:`stop_forwarders` when it pauses or is deleted.
    """

    def __init__(self) -> None:
        # sandbox_id -> list of asyncio.Server instances
        self._servers: dict[str, list[asyncio.Server]] = {}

    async def start_forwarders(
        self,
        sandbox_id: str,
        mappings: list[tuple[int, str, int]],
    ) -> None:
        """Start TCP forwarders for *sandbox_id*.

        Parameters
        ----------
        sandbox_id:
            Unique identifier (typically the container name).
        mappings:
            List of ``(local_port, target_host, target_port)`` tuples.
            A listener is created on ``0.0.0.0:local_port`` that forwards
            every accepted connection to ``target_host:target_port``.
        """
        # Stop any existing forwarders for this sandbox first
        await self.stop_forwarders(sandbox_id)

        servers: list[asyncio.Server] = []
        for local_port, target_host, target_port in mappings:
            try:
                handler = functools.partial(
                    self._handle_connection,
                    target_host=target_host,
                    target_port=target_port,
                )
                server = await asyncio.start_server(
                    handler,
                    host='0.0.0.0',
                    port=local_port,
                    reuse_address=True,
                )
                servers.append(server)
                _logger.info(
                    f'TCP forwarder listening on 0.0.0.0:{local_port} -> '
                    f'{target_host}:{target_port} (sandbox={sandbox_id})'
                )
            except OSError as exc:
                _logger.warning(
                    f'Failed to start TCP forwarder on port {local_port} '
                    f'for sandbox {sandbox_id}: {exc}'
                )

        if servers:
            self._servers[sandbox_id] = servers

    async def stop_forwarders(self, sandbox_id: str) -> None:
        """Stop and remove all TCP forwarders for *sandbox_id*."""
        servers = self._servers.pop(sandbox_id, [])
        for server in servers:
            server.close()
        for server in servers:
            await server.wait_closed()
        if servers:
            _logger.info(
                f'Stopped {len(servers)} TCP forwarder(s) for sandbox {sandbox_id}'
            )

    async def _handle_connection(
        self,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        target_host: str,
        target_port: int,
    ) -> None:
        """Pipe a single accepted connection to the target."""
        try:
            target_reader, target_writer = await asyncio.open_connection(
                target_host, target_port
            )
        except OSError as exc:
            _logger.debug(f'Cannot connect to {target_host}:{target_port}: {exc}')
            client_writer.close()
            await client_writer.wait_closed()
            return

        async def _pipe(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                while True:
                    data = await reader.read(_PIPE_BUF_SIZE)
                    if not data:
                        break
                    writer.write(data)
                    await writer.drain()
            except (ConnectionResetError, BrokenPipeError, OSError):
                pass
            finally:
                try:
                    writer.close()
                    await writer.wait_closed()
                except OSError:
                    pass

        await asyncio.gather(
            _pipe(client_reader, target_writer),
            _pipe(target_reader, client_writer),
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager: TcpPortForwarderManager | None = None


def set_tcp_port_forwarder_manager(manager: TcpPortForwarderManager) -> None:
    """Set the global TCP port forwarder manager.

    Called once during application startup when running inside a
    GitHub Codespace with a shared Docker network.
    """
    global _manager
    _manager = manager
    _logger.info('TCP port forwarder manager initialised')


def get_tcp_port_forwarder_manager() -> TcpPortForwarderManager | None:
    """Return the global manager, or ``None`` if not initialised."""
    return _manager
