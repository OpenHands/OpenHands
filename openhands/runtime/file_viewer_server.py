# IMPORTANT: LEGACY V0 CODE - Deprecated since version 1.0.0, scheduled for removal April 1, 2026
# This file is part of the legacy (V0) implementation of OpenHands and will be removed soon as we complete the migration to V1.
# OpenHands V1 uses the Software Agent SDK for the agentic core and runs a new application server. Please refer to:
#   - V1 agentic core (SDK): https://github.com/OpenHands/software-agent-sdk
#   - V1 application server (in this repo): openhands/app_server/
# Unless you are working on deprecation, please avoid extending this legacy file and consult the V1 codepaths above.
# Tag: Legacy-V0
"""A tiny, isolated server that provides only the /view endpoint from the action execution server.
This server has no authentication and only listens to localhost traffic.
"""

import os
import threading

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from uvicorn import Config, Server

from openhands.core.logger import openhands_logger as logger
from openhands.runtime.utils.file_viewer import generate_file_viewer_html


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    app = FastAPI(
        title='File Viewer Server', openapi_url=None, docs_url=None, redoc_url=None
    )

    @app.get('/')
    async def root() -> dict[str, str]:
        """Root endpoint to check if the server is running."""
        return {'status': 'File viewer server is running'}

    @app.get('/view')
    async def view_file(path: str, request: Request) -> HTMLResponse:
        """View a file using an embedded viewer.

        Args:
            path (str): The absolute path of the file to view.
            request (Request): The FastAPI request object.

        Returns:
            HTMLResponse: An HTML page with an appropriate viewer for the file.
        """
        # Security check: Only allow requests from localhost
        client_host = request.client.host if request.client else None
        if client_host not in ['127.0.0.1', 'localhost', '::1']:
            return HTMLResponse(
                content='<h1>Access Denied</h1><p>This endpoint is only accessible from localhost</p>',
                status_code=403,
            )

        if not os.path.isabs(path):
            return HTMLResponse(
                content=f'<h1>Error: Path must be absolute</h1><p>{path}</p>',
                status_code=400,
            )

        if not os.path.exists(path):
            return HTMLResponse(
                content=f'<h1>Error: File not found</h1><p>{path}</p>', status_code=404
            )

        if os.path.isdir(path):
            return HTMLResponse(
                content=f'<h1>Error: Path is a directory</h1><p>{path}</p>',
                status_code=400,
            )

        try:
            html_content = generate_file_viewer_html(path)
            return HTMLResponse(content=html_content)

        except Exception as e:
            return HTMLResponse(
                content=f'<h1>Error viewing file</h1><p>{path}</p><p>{str(e)}</p>',
                status_code=500,
            )

    return app


def start_file_viewer_server(port: int) -> tuple[str, threading.Thread]:
    """Start the file viewer server on the specified port or find an available one.

    Args:
        port (int, optional): The port to bind to. If None, an available port will be found.

    Returns:
        Tuple[str, threading.Thread]: The server URL and the thread object.
    """
    # Save the server URL to a file
    server_url = f'http://localhost:{port}'
    port_path = '/tmp/oh-server-url'
    os.makedirs(os.path.dirname(port_path), exist_ok=True)
    with open(port_path, 'w') as f:
        f.write(server_url)

    logger.info(f'File viewer server URL saved to /tmp/oh-server-url: {server_url}')
    logger.info(f'Starting file viewer server on port {port}')

    app = create_app()
    config = Config(app=app, host='127.0.0.1', port=port, log_level='error')
    server = Server(config=config)

    # Run the server in a new thread
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    return server_url, thread


if __name__ == '__main__':
    url, thread = start_file_viewer_server(port=8000)
    # Keep the main thread running
    try:
        thread.join()
    except KeyboardInterrupt:
        logger.info('Server stopped')
