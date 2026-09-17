/**
 * Renderer-side wiring for MARS port-forward tunnels (MARSOHS-1429) — the
 * registration/lifecycle half behind whatever "Add Backend" DO tile
 * eventually lands (MARSOHS-1195, tracked separately; not built here).
 *
 * Unrelated to this directory's other files (mars-client.ts, mars-launch.tsx,
 * ...): those are the older "Launch on DO MARS" PoC (MARSOHS-1194), which
 * talks to MARS directly from the browser over its own SSE event protocol and
 * deliberately does not plug into the backend registry (see mars-config.ts's
 * header comment). This module is the opposite: it exists specifically to
 * make a MARS session look like an ordinary local backend, via the
 * port-forward tunnel client built in scripts/tunnel-client.mjs +
 * tunnel-registry.mjs, running in Electron's main process.
 *
 * `window.marsBridge` (see electron/mars-preload.cjs) only exists in the
 * Electron desktop build — every export here throws in the browser/library
 * build, where that bridge cannot exist at all (no local main process to
 * hold the tunnel).
 */

import type { Backend } from "#/api/backend-registry/types";

export interface MarsTunnelStatus {
  sessionId: string;
  status: "connecting" | "connected" | "error";
  remotePort: number;
  localPort: number | undefined;
  error: string | undefined;
}

export interface OpenMarsTunnelParams {
  sessionId: string;
  remotePort: number;
  accessToken: string;
  apiUrl?: string;
  localPort?: number;
}

declare global {
  interface Window {
    marsBridge?: {
      openTunnel: (params: OpenMarsTunnelParams) => Promise<MarsTunnelStatus>;
      closeTunnel: (sessionId: string) => Promise<void>;
      getTunnel: (sessionId: string) => Promise<MarsTunnelStatus | undefined>;
    };
  }
}

function requireMarsBridge(): NonNullable<Window["marsBridge"]> {
  if (!window.marsBridge) {
    throw new Error(
      "window.marsBridge is unavailable — MARS tunnels only work in the Electron desktop build.",
    );
  }
  return window.marsBridge;
}

/** Opens (or reuses) the port-forward tunnel for one MARS session. */
export function openMarsTunnel(
  params: OpenMarsTunnelParams,
): Promise<MarsTunnelStatus> {
  return requireMarsBridge().openTunnel(params);
}

/** Closes the tunnel for one MARS session, if any. */
export function closeMarsTunnel(sessionId: string): Promise<void> {
  return requireMarsBridge().closeTunnel(sessionId);
}

/** Reads a session's current tunnel status without opening or closing it. */
export function getMarsTunnel(
  sessionId: string,
): Promise<MarsTunnelStatus | undefined> {
  return requireMarsBridge().getTunnel(sessionId);
}

/**
 * Build the Backend record for a session's now-healthy tunnel, ready to pass
 * to `useActiveBackendContext().addBackend()`. The far end really is an
 * ordinary agent-server (that's the whole point of the tunnel), so this is
 * a `kind: "local"` backend like any other — no third BackendKind needed.
 *
 * `apiKey`: whatever the guest's agent-server was configured with, if
 * anything — this module has no way to know that on its own (no session
 * config / OAuth wiring exists yet); pass it through from whatever created
 * the session, or "" if the guest requires none.
 */
export function buildMarsBackendInput({
  name,
  localPort,
  apiKey = "",
}: {
  name: string;
  localPort: number;
  apiKey?: string;
}): Omit<Backend, "id" | "connectionRevision"> {
  return {
    name,
    host: `http://127.0.0.1:${localPort}`,
    apiKey,
    kind: "local",
    authMode: "api-key",
  };
}
