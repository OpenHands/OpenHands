/**
 * `window.maritimeBridge` for the web build.
 *
 * The desktop app gets the bridge from Electron's preload; a browser gets it
 * from the Agent Canvas server instead, which hosts the same trusted-process
 * bridge and exposes it same-origin under `/maritime/rpc/<method>` (see
 * scripts/maritime-web-bridge.mjs). Every method here is one POST carrying the
 * call's arguments, so the Maritime UI runs unchanged: it only ever talks to
 * the `MaritimeBridge` interface. The Maritime API key stays on the server.
 *
 * The server wants its session key on these routes, the same key this page
 * already holds for its own agent-server, sent as `X-Session-API-Key` exactly
 * as the DigitalOcean web bridge sends it.
 */

import { getMarsBridgeHeaders } from "#/api/mars/mars-web-bridge";
import type { MaritimeBridge } from "./maritime-bridge";

export const MARITIME_WEB_HEALTH_PATH = "/maritime/health";
const RPC_PREFIX = "/maritime/rpc/";
const PROBE_TIMEOUT_MS = 1_500;

interface RpcEnvelope<T> {
  result?: T;
  error?: { message?: string; status?: number };
}

/** A bridge call the server refused; `status` is the HTTP status it chose. */
export class MaritimeWebBridgeError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "MaritimeWebBridgeError";
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
    headers: {
      "Content-Type": "application/json",
      ...getMarsBridgeHeaders(baseUrl),
    },
    body: JSON.stringify({ args }),
  });
  let envelope: RpcEnvelope<T> = {};
  try {
    envelope = (await response.json()) as RpcEnvelope<T>;
  } catch {
    // A non-JSON body (edge error page) falls through to the status line.
  }
  if (!response.ok) {
    throw new MaritimeWebBridgeError(
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
export function createMaritimeWebBridge(
  baseUrl = "",
  fetchImpl: typeof fetch = (...params) => fetch(...params),
): MaritimeBridge {
  const call =
    <T>(method: string) =>
    (...args: unknown[]) =>
      rpc<T>(baseUrl, method, args, fetchImpl);
  return {
    getAuthState: call("getAuthState"),
    saveApiKey: call("saveApiKey"),
    signOut: call("signOut"),
    getAccount: call("getAccount"),
    listAgents: call("listAgents"),
    createAgent: call("createAgent"),
    openAgent: call("openAgent"),
    resetAccessKey: call("resetAccessKey"),
    sleepAgent: call("sleepAgent"),
    deleteAgent: call("deleteAgent"),
  };
}

/** Whether the server this page came from hosts the Maritime web bridge. */
export async function probeMaritimeWebBridge(
  baseUrl = "",
  fetchImpl: typeof fetch = (...params) => fetch(...params),
  timeoutMs = PROBE_TIMEOUT_MS,
): Promise<boolean> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetchImpl(`${baseUrl}${MARITIME_WEB_HEALTH_PATH}`, {
      signal: controller.signal,
    });
    if (!response.ok) return false;
    // A server that answers every path with index.html is also a 200; only
    // the bridge says `{ ok: true }`.
    try {
      const body = (await response.json()) as { ok?: unknown };
      return body?.ok === true;
    } catch {
      return false;
    }
  } catch {
    return false;
  } finally {
    clearTimeout(timer);
  }
}

/**
 * Install the web bridge as `window.maritimeBridge` when this page is served
 * by an Agent Canvas server that hosts it. A no-op in Electron (the preload
 * has already provided the bridge) and when the server does not offer
 * Maritime, so the provider stays hidden. Resolves before the app renders
 * because `getMaritimeBridge()` is read synchronously.
 */
export async function installMaritimeWebBridge(): Promise<boolean> {
  if (typeof window === "undefined" || window.maritimeBridge) return false;
  if (!(await probeMaritimeWebBridge())) return false;
  window.maritimeBridge = createMaritimeWebBridge();
  return true;
}
