# OpenHands Legacy (V0) Server

> **IMPORTANT**: This is the legacy V0 web server, deprecated since version 1.0.0.
> The V1 application server lives under `openhands/app_server/`.

This package provides the V0 ASGI entry point (`listen.py`) that is still used by
the Makefile and the container `CMD`. It assembles the FastAPI app from
`openhands.server.app`, layers on middleware, Socket.IO, and (optionally) serves
the frontend static build.

## Key modules

| Module | Purpose |
|--------|---------|
| `app.py` | Creates the FastAPI application, mounts MCP and V1 routers |
| `listen.py` | Adds middleware (CORS, caching, rate-limiting), mounts SPA static files, wraps in Socket.IO ASGI |
| `listen_socket.py` | Re-exports the `sio` Socket.IO server for backward compatibility |
| `shared.py` | Module-level singletons (config, server config, Socket.IO, store implementations) |
| `middleware.py` | CORS, cache-control, and rate-limiting middleware |
| `static.py` | `SPAStaticFiles` — serves the frontend with SPA fallback |
| `types.py` | Shared types (`AppMode`, `ServerConfigInterface`, error classes) |
| `config/server_config.py` | OSS `ServerConfig` implementation and loader |

## Starting the server

```sh
uvicorn openhands.server.listen:app --host 0.0.0.0 --port 3000
```
