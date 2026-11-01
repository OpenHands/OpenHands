#!/usr/bin/env node
/**
 * Bounded stand-in for DigitalOcean's Managed Agents API and port-forward
 * tunnel, so a verification run can drive the Managed Agents UI without a
 * DigitalOcean account. Canvas, its bridge and the session's agent-server are
 * real; only DigitalOcean is stood in.
 *
 *   node .agents/skills/verify-openhands/scripts/fixtures/digitalocean-standin.mjs \
 *     --run "$OH_VERIFY_RUN" [--seconds 1800] [--guest-url http://127.0.0.1:PORT]
 *
 * It points the run at itself (MARS_API_BASE_URL in the run's saved launcher
 * environment), turns the web bridge on (MARS_WEB=1, which it needs) and
 * restores both on exit; `control-openhands restart` then makes Canvas use it.
 * Without --guest-url it starts the run's pinned agent-server as the sandbox:
 * no session key, its own OH_SECRET_KEY, a private HOME. Every session
 * tunnels to that one sandbox.
 *
 * The only token it accepts is the placeholder `dop_v1_` + 64 zeros, so a real
 * token is refused like a wrong one. It reads and records no other secret.
 * `<run>/private/digitalocean-standin/` holds standin.json (port, PIDs,
 * deadline), calls.jsonl (method, path, status per call) and guest.log.
 * `GET /__standin/state` returns the agents and sessions. Stop it with SIGTERM.
 */

import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import {
  appendFileSync,
  chmodSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { createServer } from "node:http";
import { connect } from "node:net";
import { join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import { WebSocketServer } from "ws";

export const PLACEHOLDER_TOKEN = `dop_v1_${"0".repeat(64)}`;
/** The guest port the bridge dials (AGENT_SERVER_GUEST_PORT in scripts/mars-api.mjs). */
export const GUEST_PORT = 8000;
export const READY = "SESSION_STATUS_READY";
export const PAUSED = "SESSION_STATUS_PAUSED";
export const PROVISIONING = "SESSION_STATUS_PROVISIONING";
/** harness-api's close codes on the port-forward socket (scripts/tunnel-client.mjs). */
export const CLOSE_SESSION_UNAVAILABLE = 4001;
export const CLOSE_GUEST_DIAL_FAILED = 4002;
const MAX_SECONDS = 7200;
const GUEST_START_MS = 180_000;

const sessionRecord = (session_id, name, config_id, status) => {
  const now = new Date().toISOString();
  return {
    session_id,
    name,
    config_id,
    status,
    agent_kind: "AGENT_KIND_UNSPECIFIED",
    created_at: now,
    last_event_at: now,
  };
};

/**
 * The seeded team: one OpenHands agent with a ready and a paused session, and
 * an agent of another kind that Canvas must not list.
 */
export function seedState() {
  const now = new Date().toISOString();
  return {
    account: {
      email: "standin@example.com",
      team: { uuid: "team-standin", name: "Stand-in team" },
    },
    configs: [
      {
        id: "cfg_standin",
        name: "standin-agent",
        updated_at: now,
        manifest: { agent: "openhands", template: "openhands" },
      },
      {
        id: "cfg_other",
        name: "other-kind-agent",
        updated_at: now,
        manifest: { agent: "claude-code" },
      },
    ],
    sessions: [
      sessionRecord("sess_ready", "standin-ready", "cfg_standin", READY),
      sessionRecord("sess_paused", "standin-paused", "cfg_standin", PAUSED),
      sessionRecord("sess_other", "other-kind-session", "cfg_other", READY),
    ],
  };
}

/** `agent:` from a flat manifest, as scripts/mars-api.mjs writes it. */
function manifestFromYaml(yaml) {
  const agent = /^agent:\s*([A-Za-z0-9._-]+)\s*$/m.exec(String(yaml ?? ""));
  return agent ? { agent: agent[1] } : {};
}

/** Configs are listed and created without their manifest, as harness-api does. */
const withoutManifest = ({ manifest: _manifest, ...config }) => config;

/**
 * The stand-in server. Exported for its tests; the CLI below adds the run
 * wiring, the sandbox and the time limit.
 *
 * @param {object} options
 * @param {string} options.guestUrl `http://127.0.0.1:<port>` of the sandbox's agent-server
 * @param {number} [options.wakeMs] How long a new or resumed session takes to be ready
 * @param {(entry: object) => void} [options.record]
 */
export function createStandin({ guestUrl, wakeMs = 2000, record = () => {} }) {
  const state = seedState();
  const guest = new URL(guestUrl);
  let nextId = 1;

  const sessionById = (id) => state.sessions.find((s) => s.session_id === id);
  const configById = (id) => state.configs.find((c) => c.id === id);
  const readyLater = (session) =>
    setTimeout(() => {
      if (sessionById(session.session_id)) session.status = READY;
    }, wakeMs).unref();
  const notFound = (what) => [
    404,
    { id: "not_found", message: `No such ${what}.` },
  ];

  /** @returns {[number, unknown]} */
  function route(method, path, query, body) {
    const page = (key, items) => [200, { [key]: items, next_page_token: "" }];
    const byStatus = (sessions) =>
      query.get("status")
        ? sessions.filter((s) => s.status === query.get("status"))
        : sessions;
    let m;
    if (method === "GET" && path === "/v2/account")
      return [200, { account: state.account }];
    if (method === "GET" && path === "/v2/agents/sessions/sandbox/sizes")
      return [
        200,
        { sizes: [{ slug: "mars-2vcpu-4gb", vcpus: 2, memory_mb: 4096 }] },
      ];
    if (path === "/v2/agents/sessions") {
      if (method === "GET") return page("sessions", byStatus(state.sessions));
      if (method === "POST") {
        if (!configById(body?.config_id)) return notFound("agent");
        const session = sessionRecord(
          `sess_new${nextId++}`,
          body.name ?? null,
          body.config_id,
          PROVISIONING,
        );
        state.sessions.push(session);
        readyLater(session);
        return [200, { session }];
      }
    }
    if (
      (m = /^\/v2\/agents\/sessions\/([^/]+)(\/pause|\/resume)?$/.exec(path))
    ) {
      const session = sessionById(m[1]);
      if (!session) return notFound("session");
      if (method === "GET" && !m[2]) return [200, { session }];
      if (method === "DELETE" && !m[2]) {
        state.sessions = state.sessions.filter((s) => s !== session);
        return [200, {}];
      }
      if (method === "POST" && m[2] === "/pause") {
        session.status = PAUSED;
        return [200, {}];
      }
      if (method === "POST" && m[2] === "/resume") {
        if (session.status === PAUSED) readyLater(session);
        return [200, {}];
      }
    }
    if (path === "/v2/agents/configs") {
      if (method === "GET")
        return page("configs", state.configs.map(withoutManifest));
      if (method === "POST") {
        const config = {
          id: `cfg_new${nextId++}`,
          name: body?.name ?? null,
          updated_at: new Date().toISOString(),
          manifest: manifestFromYaml(body?.manifest_yaml),
        };
        state.configs.push(config);
        return [200, { config: withoutManifest(config) }];
      }
    }
    if ((m = /^\/v2\/agents\/configs\/([^/]+)(\/sessions)?$/.exec(path))) {
      const config = configById(m[1]);
      if (!config) return notFound("agent");
      if (method === "GET" && m[2])
        return page(
          "sessions",
          byStatus(state.sessions.filter((s) => s.config_id === config.id)),
        );
      if (method === "GET") return [200, { config }];
      if (method === "DELETE") {
        state.configs = state.configs.filter((c) => c !== config);
        state.sessions = state.sessions.filter(
          (s) => s.config_id !== config.id,
        );
        return [200, {}];
      }
    }
    return [404, { id: "not_found", message: "Not in the stand-in." }];
  }

  const authorized = (req) =>
    req.headers.authorization === `Bearer ${PLACEHOLDER_TOKEN}`;
  const unauthorized = {
    id: "unauthorized",
    message: "Unable to authenticate you",
  };

  const server = createServer((req, res) => {
    const url = new URL(req.url, "http://standin");
    const reply = (status, payload) => {
      record({ method: req.method, path: url.pathname, status });
      res.writeHead(status, { "Content-Type": "application/json" });
      res.end(JSON.stringify(payload));
    };
    if (req.method === "GET" && url.pathname === "/__standin/state") {
      reply(200, state);
      return;
    }
    if (!authorized(req)) {
      reply(401, unauthorized);
      return;
    }
    let raw = "";
    req.on("data", (chunk) => {
      raw += chunk;
    });
    req.on("end", () => {
      let body = null;
      try {
        body = raw ? JSON.parse(raw) : null;
      } catch {
        reply(400, { id: "bad_request", message: "Body is not JSON." });
        return;
      }
      reply(...route(req.method, url.pathname, url.searchParams, body));
    });
  });

  const sockets = new WebSocketServer({ noServer: true });
  server.on("upgrade", (req, socket, head) => {
    const path = new URL(req.url, "http://standin").pathname;
    const m = /^\/v2\/agents\/sessions\/([^/]+)\/port-forward\/(\d+)$/.exec(
      path,
    );
    const tunnel = (status, extra = {}) =>
      record({ method: "TUNNEL", path, status, ...extra });
    if (!m || !authorized(req)) {
      const status = m ? 401 : 404;
      tunnel(status);
      // Node drops its own error listener when it hands over an upgrade, so a
      // client that resets before this reply would otherwise take us down.
      socket.on("error", () => socket.destroy());
      socket.end(
        `HTTP/1.1 ${status} ${m ? "Unauthorized" : "Not Found"}\r\n` +
          "Content-Type: application/json\r\nConnection: close\r\n\r\n" +
          JSON.stringify(m ? unauthorized : { id: "not_found" }),
      );
      return;
    }
    sockets.handleUpgrade(req, socket, head, (ws) => {
      // A bad frame from the client: ws already closes the tunnel (1002 and
      // the like), and the close handler below destroys the upstream.
      ws.on("error", (error) =>
        tunnel("error", { error: error.code ?? error.message }),
      );
      const session = sessionById(m[1]);
      if (session?.status !== READY) {
        tunnel(CLOSE_SESSION_UNAVAILABLE, { session: session?.status ?? null });
        ws.close(CLOSE_SESSION_UNAVAILABLE, "session unavailable");
        return;
      }
      if (Number(m[2]) !== GUEST_PORT) {
        tunnel(CLOSE_GUEST_DIAL_FAILED);
        ws.close(CLOSE_GUEST_DIAL_FAILED, "nothing listens on that port");
        return;
      }
      tunnel(101);
      const upstream = connect(Number(guest.port), guest.hostname);
      upstream.on("data", (chunk) => {
        if (ws.readyState === ws.OPEN) ws.send(chunk);
      });
      upstream.on("error", () => ws.close(CLOSE_GUEST_DIAL_FAILED, "sandbox"));
      upstream.on("close", () => ws.close());
      ws.on("message", (data) => upstream.write(data));
      ws.on("close", () => upstream.destroy());
    });
  });

  /**
   * Ends every tunnel (its close handler destroys the upstream), then the
   * server. A tunnel is an upgraded socket: closeAllConnections() does not see
   * it, and server.close() would wait for it.
   */
  const close = () => {
    for (const ws of sockets.clients) ws.terminate();
    server.closeAllConnections();
    return Promise.all([
      new Promise((resolve) => sockets.close(resolve)),
      new Promise((resolve) => server.close(resolve)),
    ]);
  };

  return { server, state, close };
}

// --- CLI: run wiring, the sandbox, the time limit -------------------------

function readOptions() {
  const { values } = parseArgs({
    options: {
      run: { type: "string" },
      seconds: { type: "string", default: "1800" },
      "guest-url": { type: "string" },
      "wake-ms": { type: "string", default: "2000" },
    },
  });
  const seconds = Number(values.seconds);
  if (!values.run) throw new Error("--run <run directory> is required");
  if (!Number.isInteger(seconds) || seconds < 1 || seconds > MAX_SECONDS)
    throw new Error(`--seconds must be within 1..${MAX_SECONDS}`);
  return { ...values, seconds, wakeMs: Number(values["wake-ms"]) };
}

const readJson = (file) => JSON.parse(readFileSync(file, "utf8"));
const writePrivate = (file, value) =>
  writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`, { mode: 0o600 });

const listen = (server, port = 0) =>
  new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", () => resolve(server.address().port));
  });

async function freePort() {
  const probe = createServer();
  const port = await listen(probe);
  await new Promise((resolve) => probe.close(resolve));
  return port;
}

const groupAlive = (pgid) => {
  try {
    process.kill(-pgid, 0);
    return true;
  } catch {
    return false;
  }
};

async function stopGroup(child) {
  if (!child || !groupAlive(child.pid)) return;
  process.kill(-child.pid, "SIGTERM");
  for (let i = 0; i < 50 && groupAlive(child.pid); i += 1) await sleep(100);
  if (groupAlive(child.pid)) process.kill(-child.pid, "SIGKILL");
}

/** The run's own agent-server command, minus Canvas's tool module. */
async function guestCommand(run, launcherEnv) {
  const { repo } = readJson(join(run, "run.json"));
  const { buildAgentServerCommand } = await import(
    pathToFileURL(join(repo, "scripts", "dev-safe.mjs")).href
  );
  const { command, args } = buildAgentServerCommand(launcherEnv);
  const at = args.indexOf("--import-modules");
  if (at !== -1) args.splice(at, 2);
  return { command, args };
}

/** @param {(child: import("node:child_process").ChildProcess) => void} onSpawn */
async function startGuest(run, dir, launcherEnv, onSpawn) {
  const home = join(dir, "guest-home");
  mkdirSync(join(home, "workspace"), { recursive: true });
  const port = await freePort();
  const { command, args } = await guestCommand(run, launcherEnv);
  const env = {
    ...launcherEnv,
    HOME: home,
    OH_SECRET_KEY: randomBytes(32).toString("hex"),
  };
  // The guest takes no key: DigitalOcean's tunnel is the gate.
  for (const name of Object.keys(env))
    if (
      /^(OH_SESSION_API_KEYS_|SESSION_API_KEY$|LOCAL_BACKEND_API_KEY$)/.test(
        name,
      )
    )
      delete env[name];
  const log = join(dir, "guest.log");
  writeFileSync(log, "", { mode: 0o600 });
  const child = spawn(
    command,
    [...args, "--host", "127.0.0.1", "--port", String(port)],
    {
      cwd: join(home, "workspace"),
      detached: true,
      env,
      stdio: ["ignore", "pipe", "pipe"],
    },
  );
  onSpawn(child);
  child.stdout.on("data", (chunk) => appendFileSync(log, chunk));
  child.stderr.on("data", (chunk) => appendFileSync(log, chunk));
  const url = `http://127.0.0.1:${port}`;
  const deadline = Date.now() + GUEST_START_MS;
  while (Date.now() < deadline && child.exitCode === null) {
    try {
      if ((await fetch(`${url}/server_info`)).ok) return { url, child };
    } catch {
      // Still starting: uvx may be resolving the pinned packages.
    }
    await sleep(1000);
  }
  await stopGroup(child);
  throw new Error(`The sandbox agent-server did not start; see ${log}`);
}

async function main() {
  const options = readOptions();
  const dir = join(options.run, "private", "digitalocean-standin");
  mkdirSync(dir, { recursive: true });
  chmodSync(dir, 0o700);
  const envFile = join(options.run, "private", "launcher-env.json");
  const launcherEnv = readJson(envFile);
  const calls = join(dir, "calls.jsonl");
  writeFileSync(calls, "", { mode: 0o600 });

  // Cleanup is in place before the guest starts: it runs in its own process
  // group, so a Ctrl-C or the deadline during a slow first start (uvx
  // resolving packages) would otherwise leave it running.
  let guestChild = null;
  let stopRunning = null;
  let stopping = false;
  const stop = async () => {
    if (stopping) return;
    stopping = true;
    if (stopRunning) await stopRunning();
    else await stopGroup(guestChild);
    process.exit(0);
  };
  process.on("SIGTERM", stop);
  process.on("SIGINT", stop);
  setTimeout(stop, options.seconds * 1000).unref();

  let guest, server, close, port;
  try {
    guest = options["guest-url"]
      ? { url: options["guest-url"], child: null }
      : await startGuest(options.run, dir, launcherEnv, (child) => {
          guestChild = child;
        });
    ({ server, close } = createStandin({
      guestUrl: guest.url,
      wakeMs: options.wakeMs,
      record: (entry) =>
        appendFileSync(
          calls,
          `${JSON.stringify({ at: new Date().toISOString(), ...entry })}\n`,
        ),
    }));
    port = await listen(server);
  } catch (error) {
    if (stopping) return; // Cancelled while starting; stop() exits.
    await stopGroup(guestChild);
    throw error;
  }
  const baseUrl = `http://127.0.0.1:${port}`;

  // The web bridge mounts only with MARS_WEB=1.
  const overrides = { MARS_API_BASE_URL: baseUrl, MARS_WEB: "1" };
  const previous = Object.fromEntries(
    Object.keys(overrides).map((name) => [name, launcherEnv[name] ?? null]),
  );
  const metadata = {
    pid: process.pid,
    baseUrl,
    guestUrl: guest.url,
    guestPgid: guest.child?.pid ?? null,
    deadline: Math.floor(Date.now() / 1000) + options.seconds,
    previousMarsApiBaseUrl: previous.MARS_API_BASE_URL,
    previousMarsWeb: previous.MARS_WEB,
    token: PLACEHOLDER_TOKEN,
  };

  stopRunning = async () => {
    // Put the run back, unless someone pointed it elsewhere meanwhile.
    const env = readJson(envFile);
    if (env.MARS_API_BASE_URL === baseUrl) {
      for (const [name, value] of Object.entries(previous))
        if (value === null) delete env[name];
        else env[name] = value;
      writePrivate(envFile, env);
    }
    await stopGroup(guest.child);
    writePrivate(join(dir, "standin.json"), {
      ...metadata,
      stoppedAt: new Date().toISOString(),
    });
    // Bounded, so a client that keeps a socket open cannot keep us running.
    await Promise.race([close(), sleep(1000)]);
  };
  writePrivate(envFile, { ...launcherEnv, ...overrides });
  writePrivate(join(dir, "standin.json"), metadata);
  guest.child?.on("exit", stop);

  console.log(
    JSON.stringify({
      ok: true,
      baseUrl,
      guestUrl: guest.url,
      seconds: options.seconds,
      next: "control-openhands restart, then browser reload; sign in with the token in standin.json",
    }),
  );
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  main().catch((error) => {
    console.error(
      JSON.stringify({ ok: false, error: String(error?.message ?? error) }),
    );
    process.exit(1);
  });
}
