/**
 * The front door to a DigitalOcean session's guest port.
 *
 * Since November 2026, DigitalOcean's `openhands` template starts the guest
 * agent-server with a session key of its own, and serves Agent Canvas
 * (scripts/static-server.mjs) on the guest port in front of it. That server
 * writes the key into the page it serves, the way a local Canvas does, so
 * the page is the one place a client can learn it.
 *
 * Every caller that reaches a session through Canvas comes through here: the
 * desktop renderer, the web build's /mars/sessions proxy, an agent logged in
 * by its first message, and the SDK. The front reads the key from the guest's
 * page once, keeps it in this process, and adds it to every request and
 * WebSocket upgrade it passes on. No page, browser storage or caller ever
 * holds it. Reaching the front still needs what reaching the guest needed
 * before guests had keys: a DigitalOcean credential to dial the tunnel, and a
 * loopback connection on this host.
 *
 * A guest whose page carries no key (an older template, where the guest port
 * is the agent-server itself) gets no key added and works as before.
 */

import { createServer, get as httpGet } from "node:http";

import { createProxyHandlers } from "./proxy-utils.mjs";
import { startPortForwardTunnel } from "./tunnel-client.mjs";

const GUEST_KEY_HEADER = "x-session-api-key";

/** A cold guest can take a while to answer its first request. */
export const GUEST_PAGE_TIMEOUT_MS = 20_000;

/** The key is written near the top of the page; this is plenty. */
const GUEST_PAGE_LIMIT_BYTES = 1024 * 1024;

/**
 * static-server.mjs writes `window.__AGENT_CANVAS_SESSION_API_KEY__=<JSON
 * string literal>;` (serializeForInlineScript), so the literal is read back
 * with JSON.parse.
 */
const PAGE_KEY_PATTERN =
  /window\.__AGENT_CANVAS_SESSION_API_KEY__\s*=\s*("(?:[^"\\\r\n]|\\.)*")/;

/**
 * The session key a Canvas page carries, or null when it carries none.
 *
 * @param {string} html
 * @returns {string | null}
 */
export function readGuestSessionKey(html) {
  const match = PAGE_KEY_PATTERN.exec(html);
  if (!match) return null;
  try {
    const key = JSON.parse(match[1]);
    return typeof key === "string" && key.trim() ? key.trim() : null;
  } catch {
    return null;
  }
}

/**
 * Read the guest's page through the tunnel and return its key.
 *
 * Resolves null when the guest answered without a key: a page without one, or
 * anything other than a page (an older guest's agent-server answers `/` with
 * its server info or a redirect). Rejects when there was no whole answer to go
 * by (a 5xx, an error, a timeout, a page cut off), so the caller asks again
 * later instead of settling on "no key".
 *
 * Plain node:http, not fetch: the tunnel is on loopback, and fetch would
 * follow a proxy set in the environment (NODE_USE_ENV_PROXY).
 *
 * @param {number} port The tunnel's loopback port
 * @param {{ timeoutMs?: number }} [options]
 * @returns {Promise<string | null>}
 */
export function fetchGuestSessionKey(
  port,
  { timeoutMs = GUEST_PAGE_TIMEOUT_MS } = {},
) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const settle = (fn, value) => {
      if (settled) return;
      settled = true;
      fn(value);
    };
    const request = httpGet(
      {
        host: "127.0.0.1",
        port,
        path: "/",
        headers: { accept: "text/html", "accept-encoding": "identity" },
        timeout: timeoutMs,
      },
      (response) => {
        const status = response.statusCode ?? 0;
        if (status >= 500) {
          response.resume();
          settle(reject, new Error(`the session's page answered ${status}`));
          return;
        }
        if (status !== 200) {
          response.resume();
          settle(resolve, null);
          return;
        }
        let html = "";
        response.setEncoding("utf8");
        response.on("data", (chunk) => {
          html += chunk;
          if (html.length > GUEST_PAGE_LIMIT_BYTES) {
            settle(resolve, readGuestSessionKey(html));
            response.destroy();
          }
        });
        response.on("end", () => settle(resolve, readGuestSessionKey(html)));
        response.on("error", (error) => settle(reject, error));
        response.on("close", () =>
          settle(reject, new Error("the session's page was cut off")),
        );
      },
    );
    request.on("timeout", () =>
      request.destroy(new Error("the session's page did not answer in time")),
    );
    request.on("error", (error) => settle(reject, error));
  });
}

/**
 * The guest's key, read once and shared by concurrent requests. A read that
 * failed is not remembered, so the next request tries again; `forget()`
 * makes the next request read the page again (after the guest refused the
 * key it had).
 *
 * @param {() => Promise<string | null>} readKey
 */
function createKeyHolder(readKey) {
  /** @type {string | null | undefined} undefined = not known yet */
  let known;
  /** @type {Promise<string | null | undefined> | null} */
  let reading = null;
  return {
    /** @returns {Promise<string | null | undefined>} undefined = could not read */
    get() {
      if (known !== undefined) return Promise.resolve(known);
      reading ??= readKey()
        .then(
          (key) => {
            known = key;
            return key;
          },
          () => undefined,
        )
        .finally(() => {
          reading = null;
        });
      return reading;
    },
    forget() {
      known = undefined;
    },
  };
}

/**
 * The caller's own key header is replaced, not passed on: only the guest's key
 * opens the guest.
 *
 * @param {import("node:http").IncomingMessage} req
 * @param {string | null | undefined} key
 */
function presentGuestKey(req, key) {
  if (key) req.headers[GUEST_KEY_HEADER] = key;
}

/**
 * Listen on `localPort` and pass everything on to the tunnel's loopback port,
 * with the guest's key added.
 *
 * @param {object} options
 * @param {number} options.upstreamPort The raw tunnel's loopback port
 * @param {() => Promise<string | null>} options.readKey
 * @param {number} [options.localPort] 0 lets the OS pick one
 * @param {string} [options.address]
 * @param {(message: string) => void} [options.log]
 * @returns {Promise<{ localPort: number, stop: () => void }>}
 */
export function startGuestFront({
  upstreamPort,
  readKey,
  localPort = 0,
  address = "127.0.0.1",
  log = (message) => console.error(`[mars-tunnel] ${message}`),
}) {
  const target = `http://127.0.0.1:${upstreamPort}`;
  const keys = createKeyHolder(readKey);
  // Agent-server calls can stay silent for minutes; the tunnel itself keeps
  // the upstream alive. The guest sees the request as the caller made it, and
  // a failed tunnel answers in the same JSON shape as the web bridge's proxy.
  const proxy = createProxyHandlers({
    label: "mars-guest",
    timeout: 0,
    proxyTimeout: 0,
    xfwd: false,
    errorBody: (message) => ({
      contentType: "application/json",
      body: JSON.stringify({
        error: { message: `upstream request failed: ${message}` },
      }),
    }),
  });

  const sockets = new Set();
  const server = createServer((req, res) => {
    keys
      .get()
      .then((key) => {
        presentGuestKey(req, key);
        // A guest that restarted mints a new key; read it again for the next
        // request rather than keep sending the old one.
        res.once("finish", () => {
          if (res.statusCode === 401) keys.forget();
        });
        proxy.proxyHttp(req, res, target);
      })
      .catch(() => res.destroy());
  });
  server.on("upgrade", (req, socket, head) => {
    socket.on("error", () => {});
    keys
      .get()
      .then((key) => {
        if (socket.destroyed) return;
        presentGuestKey(req, key);
        proxy.proxyWebSocket(req, socket, head, target);
      })
      .catch(() => socket.destroy());
  });
  server.on("connection", (socket) => {
    sockets.add(socket);
    socket.once("close", () => sockets.delete(socket));
  });
  server.on("clientError", (_error, socket) => socket.destroy());
  // A long upload must not be cut off by Node's default request timeout.
  server.requestTimeout = 0;

  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(localPort, address, () => {
      server.off("error", reject);
      server.on("error", (error) => log(`front error: ${error.message}`));
      resolve({
        localPort: server.address().port,
        stop() {
          for (const socket of sockets) socket.destroy();
          server.close();
        },
      });
    });
  });
}

/**
 * startPortForwardTunnel with the front in front of it: the raw tunnel listens
 * on a port of its own, and the snapshot's `localPort` is the front's, so it
 * is the only way in. Same options and result as startPortForwardTunnel.
 *
 * @param {Parameters<typeof startPortForwardTunnel>[0]} options
 * @param {{ startTunnel?: typeof startPortForwardTunnel, readKey?: (port: number) => Promise<string | null> }} [overrides] For tests
 */
export async function startGuestTunnel(
  { localPort = 0, address = "127.0.0.1", ...options } = {},
  { startTunnel = startPortForwardTunnel, readKey = fetchGuestSessionKey } = {},
) {
  const tunnel = await startTunnel({
    ...options,
    localPort: 0,
    address: "127.0.0.1",
  });
  let front;
  try {
    front = await startGuestFront({
      upstreamPort: tunnel.localPort,
      readKey: () => readKey(tunnel.localPort),
      localPort,
      address,
      log: options.log,
    });
  } catch (error) {
    // Keep the listen error (EADDRINUSE) for the registry's port fallback.
    tunnel.stop();
    throw error;
  }
  return {
    ...tunnel,
    localPort: front.localPort,
    stop() {
      front.stop();
      tunnel.stop();
    },
  };
}
