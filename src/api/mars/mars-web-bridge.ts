/**
 * `window.marsBridge` for the web build.
 *
 * The desktop app gets the bridge from Electron's preload; a browser gets it
 * from the Agent Canvas server instead, which hosts the same main-process
 * bridge and exposes it same-origin under `/mars/rpc/<method>` (see
 * scripts/mars-web-bridge.mjs). Every method here is one POST carrying the
 * call's arguments, so the rest of the Managed Agents UI runs unchanged: it
 * only ever talks to the `MarsBridge` interface.
 *
 * Connections resolve to a path-prefixed backend on this origin
 * (`/mars/sessions/<id>`), which the server proxies to the session's public
 * ingress URL with the DigitalOcean token added server-side. The token never
 * reaches the page.
 */

import type { MarsBridge } from "./mars-tunnel-backend";

export const MARS_WEB_HEALTH_PATH = "/mars/health";
const RPC_PREFIX = "/mars/rpc/";
const PROBE_TIMEOUT_MS = 1_500;

interface RpcEnvelope<T> {
  result?: T;
  error?: { message?: string; status?: number };
}

/** A bridge call the server refused; `status` is the HTTP status it chose. */
export class MarsWebBridgeError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "MarsWebBridgeError";
    this.status = status;
  }
}

async function rpc<T>(
  baseUrl: string,
  method: string,
  args: unknown[],
  fetchImpl: typeof fetch,
): Promise<T> {
  const response = await fetchImpl(`${baseUrl}${RPC_PREFIX}${method}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ args }),
  });
  let envelope: RpcEnvelope<T> = {};
  try {
    envelope = (await response.json()) as RpcEnvelope<T>;
  } catch {
    // A non-JSON body (edge error page) falls through to the status line.
  }
  if (!response.ok) {
    throw new MarsWebBridgeError(
      envelope.error?.message ?? `${response.status} ${response.statusText}`,
      envelope.error?.status ?? response.status,
    );
  }
  return envelope.result as T;
}

/**
 * @param baseUrl Origin (and path prefix) the Agent Canvas server is on;
 *   empty means the page's own origin.
 */
export function createMarsWebBridge(
  baseUrl = "",
  fetchImpl: typeof fetch = (...params) => fetch(...params),
): MarsBridge {
  const call =
    <T>(method: string) =>
    (...args: unknown[]) =>
      rpc<T>(baseUrl, method, args, fetchImpl);
  return {
    getAuthState: call("getAuthState"),
    // The web server never offers OAuth (no trusted loopback redirect); the
    // auth state it returns says so, and the UI hides the button.
    signInWithOAuth: async () => {
      throw new MarsWebBridgeError(
        "OAuth sign-in is only available in the desktop app; use a personal access token.",
        400,
      );
    },
    savePat: call("savePat"),
    setActiveConnection: call("setActiveConnection"),
    signOut: call("signOut"),
    listSessions: call("listSessions"),
    listAgentConfigs: call("listAgentConfigs"),
    listConfigSessions: call("listConfigSessions"),
    createOpenHandsAgent: call("createOpenHandsAgent"),
    createSession: call("createSession"),
    pauseSession: call("pauseSession"),
    resumeSession: call("resumeSession"),
    openTunnel: call("openTunnel"),
    closeTunnel: call("closeTunnel"),
    getTunnel: call("getTunnel"),
  };
}

/** Whether the server this page came from hosts the MARS web bridge. */
export async function probeMarsWebBridge(
  baseUrl = "",
  fetchImpl: typeof fetch = (...params) => fetch(...params),
  timeoutMs = PROBE_TIMEOUT_MS,
): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetchImpl(`${baseUrl}${MARS_WEB_HEALTH_PATH}`, {
      signal: controller.signal,
    });
    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Install the web bridge as `window.marsBridge` when this page is served by
 * an Agent Canvas server that hosts it. A no-op in Electron (the preload has
 * already provided the bridge) and when the server does not offer MARS, so
 * the Managed Agents UI stays hidden exactly as before. Resolves before the
 * app renders because `getMarsBridge()` is read synchronously.
 */
export async function installMarsWebBridge(): Promise<boolean> {
  if (typeof window === "undefined" || window.marsBridge) return false;
  if (!(await probeMarsWebBridge())) return false;
  window.marsBridge = createMarsWebBridge();
  return true;
}
