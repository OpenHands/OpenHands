// @vitest-environment node
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { createServer } from "node:http";
import { connect } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, test } from "vitest";
import WebSocket from "ws";
import {
  CLOSE_GUEST_DIAL_FAILED,
  CLOSE_SESSION_UNAVAILABLE,
  PAUSED,
  PLACEHOLDER_TOKEN,
  PROVISIONING,
  READY,
  createStandin,
} from "./digitalocean-standin.mjs";

const FIXTURE = fileURLToPath(
  new URL("./digitalocean-standin.mjs", import.meta.url),
);
const AUTH = { Authorization: `Bearer ${PLACEHOLDER_TOKEN}` };
const cleanups = [];
afterEach(async () => {
  while (cleanups.length) await cleanups.pop()();
});

const listen = (server) =>
  new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => resolve(server.address().port));
  });

/** A sandbox agent-server stand-in that answers /server_info. */
async function startGuest() {
  const server = createServer((req, res) => {
    res.writeHead(req.url === "/server_info" ? 200 : 404, {
      "Content-Type": "application/json",
    });
    res.end(JSON.stringify({ title: "guest", path: req.url }));
  });
  const port = await listen(server);
  cleanups.push(() => new Promise((resolve) => server.close(resolve)));
  return `http://127.0.0.1:${port}`;
}

async function start({ wakeMs = 20 } = {}) {
  const calls = [];
  const { server, close } = createStandin({
    guestUrl: await startGuest(),
    wakeMs,
    record: (entry) => calls.push(entry),
  });
  const port = await listen(server);
  cleanups.push(close);
  const base = `http://127.0.0.1:${port}`;
  const call = async (method, path, body, headers = AUTH) => {
    const response = await fetch(`${base}${path}`, {
      method,
      headers: { ...headers, "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    return { status: response.status, body: await response.json() };
  };
  return { base, call, calls };
}

const until = async (check) => {
  for (let i = 0; i < 100; i += 1) {
    if (await check()) return;
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  throw new Error("timed out");
};

test("accepts only the placeholder token, so a real one is refused like a wrong one", async () => {
  const { call } = await start();
  const realLooking = `dop_v1_${"a1".repeat(32)}`;

  assert.equal(
    (await call("GET", "/v2/agents/configs", undefined, {})).status,
    401,
  );
  assert.equal(
    (
      await call("GET", "/v2/agents/configs", undefined, {
        Authorization: `Bearer ${realLooking}`,
      })
    ).status,
    401,
  );
  const list = await call("GET", "/v2/agents/configs");
  assert.equal(list.status, 200);
  assert.deepEqual(
    list.body.configs.map((c) => [c.id, "manifest" in c]),
    [
      ["cfg_standin", false],
      ["cfg_other", false],
    ],
  );
  const one = await call("GET", "/v2/agents/configs/cfg_standin");
  assert.equal(one.body.config.manifest.agent, "openhands");
  assert.equal(
    (await call("GET", "/v2/account")).body.account.team.name,
    "Stand-in team",
  );
});

test("creates, wakes, pauses and destroys sessions the way Canvas reads them", async () => {
  const { call } = await start();
  const created = await call("POST", "/v2/agents/sessions", {
    name: "qa-new",
    config_id: "cfg_standin",
  });
  assert.equal(created.body.session.status, PROVISIONING);
  const id = created.body.session.session_id;
  const status = async () =>
    (await call("GET", `/v2/agents/sessions/${id}`)).body.session?.status;
  await until(async () => (await status()) === READY);

  await call("POST", `/v2/agents/sessions/${id}/pause`);
  assert.equal(await status(), PAUSED);
  await call("POST", `/v2/agents/sessions/${id}/resume`);
  await until(async () => (await status()) === READY);

  const sessions = await call("GET", "/v2/agents/configs/cfg_standin/sessions");
  assert.ok(sessions.body.sessions.some((s) => s.session_id === id));
  assert.equal((await call("DELETE", `/v2/agents/sessions/${id}`)).status, 200);
  assert.equal((await call("GET", `/v2/agents/sessions/${id}`)).status, 404);

  const config = await call("POST", "/v2/agents/configs", {
    name: "qa-agent",
    manifest_yaml: "agent: openhands\ntemplate: openhands\nkeep_warm: false\n",
  });
  const full = await call("GET", `/v2/agents/configs/${config.body.config.id}`);
  assert.equal(full.body.config.manifest.agent, "openhands");
});

/** Opens a tunnel; resolves with what the sandbox sent back, or the close code. */
function tunnel(base, path, headers = AUTH) {
  return new Promise((resolve) => {
    const ws = new WebSocket(`${base.replace("http", "ws")}${path}`, {
      headers,
    });
    let received = "";
    ws.on("unexpected-response", (_req, res) =>
      resolve({ httpStatus: res.statusCode }),
    );
    ws.on("open", () => {
      ws.send(
        "GET /server_info HTTP/1.1\r\nHost: guest\r\nConnection: close\r\n\r\n",
      );
    });
    ws.on("message", (data) => {
      received += data.toString();
    });
    ws.on("close", (code) => resolve({ code, received }));
  });
}

test("tunnels a ready session's guest port to the sandbox, and refuses the rest", async () => {
  const { base, calls } = await start();

  const open = await tunnel(
    base,
    "/v2/agents/sessions/sess_ready/port-forward/8000",
  );
  assert.match(open.received, /^HTTP\/1\.1 200/);
  assert.match(open.received, /"title":"guest"/);

  assert.equal(
    (await tunnel(base, "/v2/agents/sessions/sess_paused/port-forward/8000"))
      .code,
    CLOSE_SESSION_UNAVAILABLE,
  );
  assert.equal(
    (await tunnel(base, "/v2/agents/sessions/sess_ready/port-forward/3000"))
      .code,
    CLOSE_GUEST_DIAL_FAILED,
  );
  assert.equal(
    (await tunnel(base, "/v2/agents/sessions/sess_ready/port-forward/8000", {}))
      .httpStatus,
    401,
  );
  assert.deepEqual(
    calls.filter((c) => c.method === "TUNNEL").map((c) => c.status),
    [101, CLOSE_SESSION_UNAVAILABLE, CLOSE_GUEST_DIAL_FAILED, 401],
  );
});

/** Runs the CLI against a fresh run directory; resolves once it is listening. */
async function startCli(launcherEnv = {}) {
  const run = mkdtempSync(join(tmpdir(), "oh-do-standin-"));
  mkdirSync(join(run, "private"));
  const envFile = join(run, "private", "launcher-env.json");
  writeFileSync(
    envFile,
    JSON.stringify({ PATH: process.env.PATH, ...launcherEnv }),
  );
  const child = spawn(
    process.execPath,
    [
      FIXTURE,
      "--run",
      run,
      "--guest-url",
      await startGuest(),
      "--seconds",
      "60",
    ],
    { stdio: ["ignore", "pipe", "inherit"] },
  );
  cleanups.push(() => child.exitCode === null && child.kill("SIGKILL"));
  const ready = await new Promise((resolve) => {
    child.stdout.once("data", (chunk) => resolve(JSON.parse(String(chunk))));
  });
  return { run, envFile, child, ready };
}

/** SIGTERMs the CLI; resolves with its exit code, or null after 5 s. */
function stopCli(child) {
  child.kill("SIGTERM");
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(null), 5000);
    child.once("exit", (code) => {
      clearTimeout(timer);
      resolve(code);
    });
  });
}

const openTunnel = async (baseUrl) => {
  const ws = new WebSocket(
    `${baseUrl.replace("http", "ws")}/v2/agents/sessions/sess_ready/port-forward/8000`,
    { headers: AUTH },
  );
  cleanups.push(() => ws.terminate());
  await new Promise((resolve, reject) => {
    ws.once("open", resolve);
    ws.on("error", reject);
  });
  return ws;
};

test("points the run at itself while it runs, and puts the run back when stopped", async () => {
  const { run, envFile, child, ready } = await startCli({ KEEP: "me" });

  assert.equal(ready.ok, true);
  const running = JSON.parse(readFileSync(envFile, "utf8"));
  assert.equal(running.MARS_API_BASE_URL, ready.baseUrl);
  assert.equal(running.MARS_WEB, "1");
  const metadata = join(run, "private", "digitalocean-standin", "standin.json");
  assert.equal(statSync(metadata).mode & 0o777, 0o600);
  assert.equal((await fetch(`${ready.baseUrl}/__standin/state`)).status, 200);

  assert.equal(await stopCli(child), 0);
  const env = JSON.parse(readFileSync(envFile, "utf8"));
  assert.equal(env.MARS_API_BASE_URL, undefined);
  assert.equal(env.MARS_WEB, undefined);
  assert.equal(env.KEEP, "me");
  assert.ok(JSON.parse(readFileSync(metadata, "utf8")).stoppedAt);
});

test("puts back the run's own MARS_API_BASE_URL and MARS_WEB when stopped", async () => {
  const { run, envFile, child, ready } = await startCli({
    MARS_API_BASE_URL: "https://api.example.test",
    MARS_WEB: "0",
  });

  const running = JSON.parse(readFileSync(envFile, "utf8"));
  assert.equal(running.MARS_API_BASE_URL, ready.baseUrl);
  assert.equal(running.MARS_WEB, "1");
  const metadata = join(run, "private", "digitalocean-standin", "standin.json");
  const saved = JSON.parse(readFileSync(metadata, "utf8"));
  assert.equal(saved.previousMarsApiBaseUrl, "https://api.example.test");
  assert.equal(saved.previousMarsWeb, "0");

  assert.equal(await stopCli(child), 0);
  const env = JSON.parse(readFileSync(envFile, "utf8"));
  assert.equal(env.MARS_API_BASE_URL, "https://api.example.test");
  assert.equal(env.MARS_WEB, "0");
});

test("leaves the run alone when stopped after someone pointed it elsewhere", async () => {
  const { envFile, child } = await startCli();
  const elsewhere = {
    ...JSON.parse(readFileSync(envFile, "utf8")),
    MARS_API_BASE_URL: "https://api.example.test",
  };
  writeFileSync(envFile, JSON.stringify(elsewhere));

  assert.equal(await stopCli(child), 0);
  const env = JSON.parse(readFileSync(envFile, "utf8"));
  assert.equal(env.MARS_API_BASE_URL, "https://api.example.test");
  assert.equal(env.MARS_WEB, "1");
});

test("stops on SIGTERM while a tunnel is open", async () => {
  const { envFile, child, ready } = await startCli();
  await openTunnel(ready.baseUrl);

  assert.notEqual(
    await stopCli(child),
    null,
    "the stand-in is still running 5 s after SIGTERM",
  );
  assert.equal(
    JSON.parse(readFileSync(envFile, "utf8")).MARS_API_BASE_URL,
    undefined,
  );
});

test("survives a bad frame on a tunnel, and still puts the run back when stopped", async () => {
  const { envFile, child, ready } = await startCli();
  const ws = await openTunnel(ready.baseUrl);
  const closed = new Promise((resolve) => ws.once("close", resolve));
  // A final frame with opcode 3, which WebSocket reserves.
  ws._socket.write(Buffer.from([0x83, 0x00]));

  assert.equal(await closed, 1002);
  assert.equal(child.exitCode, null);
  assert.equal((await fetch(`${ready.baseUrl}/__standin/state`)).status, 200);
  (await openTunnel(ready.baseUrl)).close();
  assert.equal(await stopCli(child), 0);
  assert.equal(
    JSON.parse(readFileSync(envFile, "utf8")).MARS_API_BASE_URL,
    undefined,
  );
});

test("survives clients that reset as soon as they ask for a refused tunnel", async () => {
  const { child, ready } = await startCli();
  const { port } = new URL(ready.baseUrl);
  const requests = [
    ["/v2/agents/sessions/sess_ready/no-such-path", AUTH],
    ["/v2/agents/sessions/sess_ready/port-forward/8000", {}],
  ];
  // Whether the reset lands before the stand-in's reply is a race; try a few.
  for (let i = 0; i < 10 && child.exitCode === null; i += 1) {
    const [path, headers] = requests[i % requests.length];
    const socket = connect(Number(port), "127.0.0.1");
    socket.on("error", () => {});
    // An error here means the stand-in is gone; the assertion below says so.
    await new Promise((resolve) => {
      socket.once("connect", resolve);
      socket.once("error", resolve);
    });
    socket.write(
      [
        `GET ${path} HTTP/1.1`,
        "Host: standin",
        "Connection: Upgrade",
        "Upgrade: websocket",
        "Sec-WebSocket-Version: 13",
        `Sec-WebSocket-Key: ${Buffer.alloc(16).toString("base64")}`,
        ...Object.entries(headers).map(([name, value]) => `${name}: ${value}`),
        "",
        "",
      ].join("\r\n"),
    );
    socket.resetAndDestroy();
    await new Promise((resolve) => setTimeout(resolve, 50));
  }

  assert.equal(child.exitCode, null);
  assert.equal((await fetch(`${ready.baseUrl}/__standin/state`)).status, 200);
});

test("stops a guest that is still starting when cancelled", async () => {
  const run = mkdtempSync(join(tmpdir(), "oh-do-standin-"));
  mkdirSync(join(run, "private"));
  mkdirSync(join(run, "repo", "scripts"), { recursive: true });
  writeFileSync(
    join(run, "private", "launcher-env.json"),
    JSON.stringify({ PATH: process.env.PATH }),
  );
  writeFileSync(
    join(run, "run.json"),
    JSON.stringify({ repo: join(run, "repo") }),
  );
  // A guest that never answers /server_info, like uvx still resolving.
  const pidFile = join(run, "guest.pid");
  const guest = join(run, "repo", "slow-guest.mjs");
  writeFileSync(
    guest,
    `import { writeFileSync } from "node:fs";
writeFileSync(${JSON.stringify(pidFile)}, String(process.pid));
setInterval(() => {}, 1000);
`,
  );
  writeFileSync(
    join(run, "repo", "scripts", "dev-safe.mjs"),
    `export const buildAgentServerCommand = () => ({
  command: ${JSON.stringify(process.execPath)},
  args: [${JSON.stringify(guest)}],
});
`,
  );
  const child = spawn(process.execPath, [FIXTURE, "--run", run], {
    stdio: "ignore",
  });
  cleanups.push(() => child.exitCode === null && child.kill("SIGKILL"));
  const alive = (pid) => {
    try {
      process.kill(pid, 0);
      return true;
    } catch {
      return false;
    }
  };
  let pid = 0;
  await until(() => {
    try {
      pid = Number(readFileSync(pidFile, "utf8"));
    } catch {
      // Not spawned yet.
    }
    return pid > 0;
  });
  cleanups.push(() => alive(pid) && process.kill(pid, "SIGKILL"));

  const exited = new Promise((resolve) => child.once("exit", resolve));
  child.kill("SIGINT");
  await exited;

  await until(() => !alive(pid));
});
