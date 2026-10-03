# Attaching to an existing Agent Server

## LA-001: Explicit attachment preserves server ownership

Setting `OH_CANVAS_ATTACH_EXISTING_AGENT_SERVER=1` reuses the Agent Server on
`127.0.0.1` at `OH_CANVAS_SAFE_BACKEND_PORT` or the configured default port.
Canvas starts its other services normally. Without this opt-in, an occupied
backend port remains an error.

The existing session key comes from `LOCAL_BACKEND_API_KEY` or the existing
`OH_SESSION_API_KEY_PATH` file, falling back to the normal persisted key path.
Attachment never generates credentials for the external server, starts a second
Agent Server, seeds secrets, clears conversation leases or signals the external
process. Workspace paths and bundled editor routing are not inferred for a
server Canvas did not start. `VITE_WORKING_DIR` can explicitly select a directory
the attached server can access.

The mode works with the full development stack, static stack, minimal stack and
published binary. It cannot be combined with `--frontend-only`.

## LA-002: Verify an attached server before starting services

Before starting any child service, Canvas checks liveness, authenticates through
the read-only settings endpoint and verifies the version reported by
`/server_info` against `compatibility.minimumAgentServer`. A missing key,
unreachable server, failed authentication or unsupported/unknown version fails
startup without a replacement server or state mutation. Each request has a
five-second timeout. Diagnostics do not include keys or server response bodies.
