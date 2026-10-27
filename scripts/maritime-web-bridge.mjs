/**
 * The Maritime provider for the web build: the Agent Canvas server stands in
 * for Electron's main process.
 *
 * In the desktop app the renderer reaches Maritime through
 * `window.maritimeBridge` (electron/preload-main.cjs →
 * scripts/maritime-bridge.mjs). In a browser the Maritime API key must still
 * stay off the page, so the server that fronts the web build
 * (scripts/ingress.mjs, scripts/static-server.mjs) hosts the same bridge and
 * exposes it same-origin:
 *
 *   POST /maritime/rpc/<method>  the MaritimeBridge surface over JSON; the
 *                                renderer's fetch-backed bridge
 *                                (src/api/maritime/maritime-web-bridge.ts)
 *                                calls it exactly as the preload calls IPC.
 *   GET  /maritime/health        lets the renderer detect this server.
 *
 * Unlike the DigitalOcean bridge there is no proxy: each Maritime agent has
 * its own public HTTPS hostname, which the browser reaches directly.
 *
 * Trust model, as for the DigitalOcean bridge: the routes hold the Maritime
 * API key and can delete agents, so every RPC must carry the server's session
 * key as `X-Session-API-Key` (`MARITIME_WEB_KEY`, else the server's
 * `--session-api-key`). With neither, the bridge mounts only on a loopback
 * bind, where a `Host` check stops DNS rebinding. Cross-origin requests are
 * refused. The integration is opt-in with `MARITIME_WEB=1`.
 *
 * Secrets survive a restart: they are kept under the Canvas state directory
 * (`MARITIME_STATE_DIR`, else `OH_CANVAS_SAFE_STATE_DIR`, else
 * `~/.openhands/agent-canvas`) in the OS keychain, a file encrypted with
 * `OH_SECRET_KEY`, or memory (scripts/mars-secret-store.mjs).
 * `MARITIME_API_KEY` signs the bridge in at startup.
 */

import { homedir } from "node:os";
import path from "node:path";

import { DEFAULT_BIND_HOST, isLoopbackBind } from "./bind-host.mjs";
import { MARITIME_IPC, createMaritimeBridge } from "./maritime-bridge.mjs";
import { openSecretStore } from "./mars-secret-store.mjs";
import {
  isCrossOrigin,
  isForeignHost,
  readJsonBody,
  safeEqual,
  writeJson,
} from "./web-bridge-guards.mjs";

export const MARITIME_WEB_PREFIX = "/maritime";
const HEALTH_PATH = `${MARITIME_WEB_PREFIX}/health`;
const RPC_PREFIX = `${MARITIME_WEB_PREFIX}/rpc/`;
const RPC_BODY_LIMIT_BYTES = 64 * 1024;
const AUTH_HEADER = "x-session-api-key";
const STATE_SUBDIR = "maritime";

/** Bridge methods a browser may call: the desktop IPC surface, no more. */
export const RPC_METHODS = new Set(Object.keys(MARITIME_IPC));

/**
 * The key `/maritime/*` requires, or null when none is configured.
 *
 * @param {object} [options]
 * @param {Record<string, string | undefined>} [options.env]
 * @param {string | null} [options.sessionApiKey]
 */
export function resolveMaritimeWebKey({
  env = process.env,
  sessionApiKey = null,
} = {}) {
  const explicit = env.MARITIME_WEB_KEY?.trim();
  if (explicit) return explicit;
  const own = typeof sessionApiKey === "string" ? sessionApiKey.trim() : "";
  return own || null;
}

/** The Canvas state directory, as the DigitalOcean bridge resolves it. */
export function resolveMaritimeStateDir(env = process.env) {
  const configured =
    env.MARITIME_STATE_DIR?.trim() || env.OH_CANVAS_SAFE_STATE_DIR?.trim();
  if (configured) return configured;
  try {
    return path.join(homedir(), ".openhands", "agent-canvas");
  } catch {
    // No home folder (`docker run --user <uid>`): secrets stay in memory.
    return null;
  }
}

/** `http(s)://host[:port]` of the request as the browser sees it. */
function requestOrigin(req) {
  const proto =
    req.headers["x-forwarded-proto"]?.split(",")[0].trim() || "http";
  return `${proto}://${req.headers.host}`;
}

/**
 * Mount only with `MARITIME_WEB=1`, and only behind a key unless the bind is
 * loopback.
 *
 * @param {object} [options]
 * @param {Record<string, string | undefined>} [options.env]
 * @param {string} [options.host]
 * @param {string | null} [options.sessionApiKey]
 * @param {(message: string) => void} [options.log]
 */
export function mountMaritimeWebBridge({
  env = process.env,
  host = DEFAULT_BIND_HOST,
  sessionApiKey = null,
  log = (message) => console.warn(`[maritime-web] ${message}`),
} = {}) {
  if (env.MARITIME_WEB !== "1") {
    const value = env.MARITIME_WEB ?? "";
    if (value && value.trim() !== "0") {
      log(
        `MARITIME_WEB=${JSON.stringify(value)} is not "1"; not mounting /maritime/* (set MARITIME_WEB=1).`,
      );
    } else if (!value && env.MARITIME_API_KEY?.trim()) {
      log(
        `MARITIME_API_KEY is set but MARITIME_WEB is not "1"; not mounting /maritime/*.`,
      );
    }
    return null;
  }
  const key = resolveMaritimeWebKey({ env, sessionApiKey });
  if (!key && !isLoopbackBind(host)) {
    log(
      `no key and bind host ${host} is not loopback; not mounting /maritime/* (set MARITIME_WEB_KEY or --session-api-key).`,
    );
    return null;
  }
  return createMaritimeWebBridge({
    env,
    host,
    key,
    stateDir: resolveMaritimeStateDir(env),
  });
}

/**
 * @param {object} [options]
 * @param {ReturnType<typeof createMaritimeBridge>} [options.bridge] Override for tests
 * @param {Record<string, string | undefined>} [options.env]
 * @param {string} [options.host] The server's bind host; loopback gets the Host check
 * @param {string | null} [options.key] Session key every RPC must present; null = open (loopback only)
 * @param {string | null} [options.stateDir]
 * @param {(message: string, error?: unknown) => void} [options.log]
 */
export function createMaritimeWebBridge({
  bridge: bridgeOverride,
  env = process.env,
  host = DEFAULT_BIND_HOST,
  key = null,
  stateDir = null,
  log = (message, error) =>
    console.warn(`[maritime-web] ${message}`, error ?? ""),
} = {}) {
  const dir = stateDir ? path.join(stateDir, STATE_SUBDIR) : null;
  const bridge =
    bridgeOverride ??
    createMaritimeBridge({
      userDataPath: dir,
      secrets: openSecretStore({ env, dir }),
      env,
    });

  if (env.MARITIME_API_KEY?.trim()) {
    bridge
      .saveApiKey({ apiKey: env.MARITIME_API_KEY })
      .catch((error) => log("MARITIME_API_KEY was not accepted", error));
  }

  function isAuthorized(req) {
    if (!key) return true;
    const presented = req.headers[AUTH_HEADER];
    return typeof presented === "string" && safeEqual(presented, key);
  }

  /**
   * Methods that differ from the bridge's own, called with the request
   * first. A new agent's CORS origin is the one this request came from,
   * not whatever the page claims.
   */
  const overrides = {
    createAgent(req, input) {
      return bridge.createAgent({
        ...(input ?? {}),
        origin: requestOrigin(req),
      });
    },
  };

  async function handleRpc(req, res, method) {
    if (req.method !== "POST") {
      writeJson(res, 405, { error: { message: "use POST" } });
      return;
    }
    if (!RPC_METHODS.has(method)) {
      writeJson(res, 404, { error: { message: `unknown method ${method}` } });
      return;
    }
    let args;
    try {
      const body = await readJsonBody(req, RPC_BODY_LIMIT_BYTES);
      args = Array.isArray(body.args) ? body.args : [];
    } catch (error) {
      writeJson(res, 400, { error: { message: error.message } });
      return;
    }
    try {
      const result = Object.hasOwn(overrides, method)
        ? await overrides[method](req, ...args)
        : await bridge[method](...args);
      writeJson(res, 200, { result: result ?? null });
    } catch (error) {
      const status =
        typeof error?.status === "number" && error.status >= 400
          ? error.status
          : 500;
      writeJson(res, status, {
        error: { message: error?.message ?? String(error), status },
      });
    }
  }

  return {
    bridge,

    /** @returns {boolean} true when the request was a Maritime request and handled */
    handleHttp(req, res) {
      const url = req.url ?? "/";
      if (!url.startsWith(`${MARITIME_WEB_PREFIX}/`)) return false;
      if (isCrossOrigin(req) || isForeignHost(req, host)) {
        writeJson(res, 403, {
          error: { message: "cross-origin request refused" },
        });
        return true;
      }
      const pathOnly = url.split(/[?#]/)[0];
      if (pathOnly === HEALTH_PATH) {
        // Unauthenticated on purpose: it only says the bridge is here.
        writeJson(res, 200, { ok: true, authRequired: key !== null });
        return true;
      }
      if (!isAuthorized(req)) {
        writeJson(res, 401, { error: { message: "session key required" } });
        return true;
      }
      if (pathOnly.startsWith(RPC_PREFIX)) {
        void handleRpc(req, res, pathOnly.slice(RPC_PREFIX.length));
        return true;
      }
      writeJson(res, 404, { error: { message: "not found" } });
      return true;
    },
  };
}
