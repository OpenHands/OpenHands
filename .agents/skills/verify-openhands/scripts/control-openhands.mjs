#!/usr/bin/env node
// control-openhands: drive a real, isolated Agent Canvas stack the way a user
// does, so every feature-map recipe is a command an agent can rerun.
//
// Run `control-openhands --help` or `control-openhands <command> --help`.

import { spawn, spawnSync, execFileSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import {
  appendFileSync,
  existsSync,
  lstatSync,
  mkdirSync,
  openSync,
  readFileSync,
  readdirSync,
  realpathSync,
  rmSync,
  symlinkSync,
  unlinkSync,
  writeFileSync,
} from "node:fs";
import { createConnection } from "node:net";
import { freemem, tmpdir } from "node:os";
import { basename, dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { deflateSync, gunzipSync } from "node:zlib";
import { buildIdentity } from "./lib/build-id.mjs";
import {
  editorOffset,
  launcherEnvFor,
  portsAfterRestart,
  runPorts,
} from "./lib/launcher-env.mjs";
import { baselineLines, parseBaseline, withBaseline } from "./lib/baseline.mjs";
import { affectedFamilies, familyHead } from "./lib/map-sources.mjs";
import { routePattern } from "./lib/route-pattern.mjs";
import {
  citedTestids,
  resolveTestids,
  sourceLiterals,
} from "./lib/testids.mjs";
import { tmuxPathFor } from "./lib/tmux-path.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const skillDir = resolve(here, "..");
const repoRoot = resolve(here, "../../../..");
const mapDir = join(skillDir, "references", "feature-map");
const verifyHome = resolve(
  process.env.OH_VERIFY_HOME || join(tmpdir(), "openhands-verify"),
);

// ---------------------------------------------------------------------------
// Output helpers. Every command prints one JSON object on stdout; exit code 0
// means success, 1 an action/assertion failure, 2 bad usage, 3 an environment
// or doctor failure.
// ---------------------------------------------------------------------------
class CliError extends Error {
  constructor(message, { code = 1, hint, extra } = {}) {
    super(message);
    this.code = code;
    this.hint = hint;
    this.extra = extra;
  }
}

function out(value) {
  process.stdout.write(`${JSON.stringify(value, null, 2)}\n`);
}

function usage(message, example) {
  throw new CliError(message, { code: 2, hint: example });
}

function parseArgs(argv) {
  const positional = [];
  const flags = {};
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--") {
      positional.push(...argv.slice(i + 1));
      break;
    }
    if (arg.startsWith("--")) {
      const eq = arg.indexOf("=");
      if (eq > 0) {
        flags[arg.slice(2, eq)] = arg.slice(eq + 1);
      } else {
        const key = arg.slice(2);
        const next = argv[i + 1];
        if (next === undefined || next.startsWith("--")) flags[key] = true;
        else {
          // --artifact may repeat; every other flag keeps its last value.
          flags[key] =
            key === "artifact" && flags[key] !== undefined
              ? [].concat(flags[key], next)
              : next;
          i += 1;
        }
      }
    } else {
      positional.push(arg);
    }
  }
  return { positional, flags };
}

function intFlag(value, fallback) {
  if (value === undefined || value === true) return fallback;
  const n = Number(value);
  if (!Number.isFinite(n)) usage(`Expected a number, got ${value}`);
  return n;
}

const delay = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------------
// Run directory handling.
// ---------------------------------------------------------------------------
function liveRuns() {
  if (!existsSync(verifyHome)) return [];
  return readdirSync(verifyHome)
    .filter((name) => name !== "current" && !name.startsWith("."))
    .map((name) => join(verifyHome, name))
    .filter((dir) => {
      try {
        const run = JSON.parse(readFileSync(join(dir, "run.json"), "utf8"));
        return groupAlive(run.launcherPgid);
      } catch {
        return false;
      }
    });
}

function resolveRunDir(flags, { required = true } = {}) {
  const explicit = flags.run || process.env.OH_VERIFY_RUN;
  let dir;
  if (explicit) dir = resolve(explicit);
  else {
    // With several live runs (several agents on one machine), guessing would
    // let one agent drive another's browser. Refuse instead.
    const live = liveRuns();
    if (live.length > 1) {
      throw new CliError(
        `${live.length} runs are alive; refusing to guess which one is yours.`,
        {
          code: 3,
          hint: `export OH_VERIFY_RUN=<your run dir> (or pass --run). Live runs: ${live.join(", ")}`,
        },
      );
    }
    if (live.length === 1) dir = live[0];
    else if (existsSync(join(verifyHome, "current"))) {
      dir = realpathSync(join(verifyHome, "current"));
    }
  }
  if (!dir || !existsSync(join(dir, "run.json"))) {
    if (!required) return undefined;
    throw new CliError("No verification run found.", {
      code: 3,
      hint: "Start one with `control-openhands launch`, or pass --run <dir> / export OH_VERIFY_RUN.",
    });
  }
  return dir;
}

function loadRun(flags, options) {
  const dir = resolveRunDir(flags, options);
  if (!dir) return undefined;
  const run = JSON.parse(readFileSync(join(dir, "run.json"), "utf8"));
  run.dir = dir;
  return run;
}

function saveRun(run) {
  const { dir, ...rest } = run;
  writeFileSync(join(dir, "run.json"), `${JSON.stringify(rest, null, 2)}\n`);
}

function sessionKey(run) {
  return readFileSync(join(run.dir, "private", "session-key"), "utf8").trim();
}

function git(args, cwd = repoRoot) {
  return execFileSync("git", args, { cwd, encoding: "utf8" }).trim();
}

function checkoutRevision() {
  return git(["rev-parse", "HEAD"]);
}

// Identity of the frontend build inputs, untracked new files included:
// commits that only touch other paths (docs, this skill) reuse the existing
// build instead of rebuilding under stacks that are already serving it.
function buildId() {
  return buildIdentity(repoRoot);
}

function processAlive(pid) {
  if (!pid) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch (error) {
    return error.code === "EPERM";
  }
}

function groupAlive(pgid) {
  if (!pgid) return false;
  try {
    process.kill(-pgid, 0);
    return true;
  } catch (error) {
    return error.code === "EPERM";
  }
}

function commandLine(pid) {
  try {
    if (existsSync(`/proc/${pid}/cmdline`)) {
      return readFileSync(`/proc/${pid}/cmdline`, "utf8")
        .replace(/\0/g, " ")
        .trim();
    }
    return execFileSync("ps", ["-o", "command=", "-p", String(pid)], {
      encoding: "utf8",
    }).trim();
  } catch {
    return "";
  }
}

function portOpen(port, host = "127.0.0.1") {
  return new Promise((resolvePromise) => {
    const socket = createConnection({ port, host });
    const done = (value) => {
      socket.destroy();
      resolvePromise(value);
    };
    socket.setTimeout(800, () => done(false));
    socket.once("connect", () => done(true));
    socket.once("error", () => done(false));
  });
}

function claimPath(base) {
  return join(verifyHome, `.ports-${base}.claim`);
}

function tryClaim(base, runDir) {
  // Concurrent launches must not pick the same block: claim it atomically
  // ('wx'), and take over only claims whose run is gone or never started.
  const path = claimPath(base);
  try {
    writeFileSync(path, runDir, { flag: "wx" });
    return true;
  } catch {
    try {
      const owner = readFileSync(path, "utf8").trim();
      const ownerRun = join(owner, "run.json");
      const alive =
        existsSync(ownerRun) &&
        groupAlive(JSON.parse(readFileSync(ownerRun, "utf8")).launcherPgid);
      const age = Date.now() - lstatSync(path).mtimeMs;
      if (alive || (!existsSync(ownerRun) && age < 10 * 60_000)) return false;
      writeFileSync(path, runDir);
      return true;
    } catch {
      return false;
    }
  }
}

function releaseClaim(run) {
  const path = claimPath(run.ports.ingress);
  try {
    if (readFileSync(path, "utf8").trim() === run.dir) unlinkSync(path);
  } catch {
    // already released
  }
}

async function findPortBlock(start, runDir, editor) {
  // The launcher needs ingress P, agent-server P+1, automation P+2, static
  // frontend P+3, and the editor port only when it serves VS Code (`editor`
  // is its offset from the agent-server port, or null; see editorOffset).
  mkdirSync(verifyHome, { recursive: true });
  for (let base = start; base < start + 900; base += 10) {
    const ports = Object.values(runPorts(base, editor));
    const busy = await Promise.all(ports.map((p) => portOpen(p)));
    if (busy.some(Boolean)) continue;
    if (tryClaim(base, runDir)) return base;
  }
  throw new CliError(`No free port block found from ${start}`, { code: 3 });
}

async function http(
  run,
  method,
  path,
  { body, raw, auth = true, timeout = 15_000, headers = {} } = {},
) {
  const url = new URL(path, run.baseUrl);
  if (url.origin !== new URL(run.baseUrl).origin) {
    throw new CliError(
      `Refusing to send the session key outside ${run.baseUrl}`,
      { code: 2 },
    );
  }
  const response = await fetch(url, {
    method,
    redirect: "manual",
    signal: AbortSignal.timeout(timeout),
    headers: {
      ...(auth ? { "X-Session-API-Key": sessionKey(run) } : {}),
      ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
      ...headers,
    },
    body: raw ?? (body === undefined ? undefined : JSON.stringify(body)),
  });
  const text = await response.text();
  let json;
  try {
    json = text ? JSON.parse(text) : undefined;
  } catch {
    json = undefined;
  }
  return {
    status: response.status,
    ok: response.ok,
    json,
    text,
    contentType: response.headers.get("content-type"),
  };
}

function defaults() {
  return JSON.parse(
    readFileSync(join(repoRoot, "config", "defaults.json"), "utf8"),
  );
}

function availableMemoryBytes() {
  // MemAvailable counts reclaimable cache; os.freemem() alone understates it.
  try {
    const line = readFileSync("/proc/meminfo", "utf8").match(
      /^MemAvailable:\s+(\d+) kB/m,
    );
    if (line) return Number(line[1]) * 1024;
  } catch {
    // not Linux
  }
  return freemem();
}

function nodeMajor() {
  return Number(process.versions.node.split(".")[0]);
}

// ---------------------------------------------------------------------------
// Browser daemon client.
// ---------------------------------------------------------------------------
function daemonInfo(run) {
  const path = join(run.dir, "private", "browser.json");
  if (!existsSync(path)) return undefined;
  const info = JSON.parse(readFileSync(path, "utf8"));
  return processAlive(info.pid) ? info : undefined;
}

async function browserCall(run, cmd, args = {}, { timeout = 120_000 } = {}) {
  const info = daemonInfo(run);
  if (!info) {
    throw new CliError("The browser daemon is not running for this run.", {
      code: 3,
      hint: "Run `control-openhands browser start`.",
    });
  }
  const response = await fetch(`http://127.0.0.1:${info.port}/`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-control-token": info.token,
    },
    body: JSON.stringify({ cmd, args }),
    signal: AbortSignal.timeout(timeout),
  });
  const result = await response.json();
  if (!result.ok) {
    throw new CliError(result.error, {
      code: 1,
      hint: result.hint,
      extra: { url: result.url, failureScreenshot: result.failureScreenshot },
    });
  }
  delete result.ok;
  return result;
}

async function startBrowser(run) {
  const existing = daemonInfo(run);
  if (existing) return { alreadyRunning: true, pid: existing.pid };
  const logFd = openSync(join(run.dir, "private", "browser.log"), "a", 0o600);
  const child = spawn(process.execPath, [join(here, "browser-daemon.mjs")], {
    cwd: repoRoot,
    detached: true,
    stdio: ["ignore", logFd, logFd],
    env: {
      ...process.env,
      OH_VERIFY_RUN: run.dir,
      OH_VERIFY_BASE_URL: run.baseUrl,
    },
  });
  child.unref();
  const browserJson = join(run.dir, "private", "browser.json");
  for (let i = 0; i < 60; i += 1) {
    await delay(500);
    if (existsSync(browserJson)) {
      const info = JSON.parse(readFileSync(browserJson, "utf8"));
      if (info.pid === child.pid) {
        run.browserPid = child.pid;
        saveRun(run);
        return { started: true, pid: child.pid };
      }
    }
    if (!processAlive(child.pid)) break;
  }
  const log = readFileSync(join(run.dir, "private", "browser.log"), "utf8")
    .split("\n")
    .slice(-15)
    .join("\n");
  throw new CliError("Browser daemon failed to start.", {
    code: 3,
    hint: "Set CONTROL_OPENHANDS_BROWSER to a Chromium executable when Playwright's pinned browser is missing.",
    extra: { log },
  });
}

async function stopBrowser(run) {
  const info = daemonInfo(run);
  if (!info) return { running: false };
  try {
    await browserCall(run, "shutdown", {}, { timeout: 5_000 });
  } catch {
    // fall through to signal
  }
  for (let i = 0; i < 20 && processAlive(info.pid); i += 1) await delay(250);
  if (processAlive(info.pid)) process.kill(info.pid, "SIGTERM");
  rmSync(join(run.dir, "private", "browser.json"), { force: true });
  return { stopped: true, pid: info.pid };
}

function versionAtLeast(version, minimum) {
  const parts = (v) =>
    String(v)
      .replace(/^v/, "")
      .split(/[.+-]/)
      .slice(0, 3)
      .map((n) => Number.parseInt(n, 10) || 0);
  const [a, b] = [parts(version), parts(minimum)];
  for (let i = 0; i < 3; i += 1) if (a[i] !== b[i]) return a[i] > b[i];
  return true;
}

// ---------------------------------------------------------------------------
// Commands: lifecycle.
// ---------------------------------------------------------------------------
const PASS_ENV = [
  "PATH",
  "LANG",
  "LC_ALL",
  "TERM",
  "TZ",
  "TMPDIR",
  "USER",
  "LOGNAME",
  "SHELL",
  "HTTP_PROXY",
  "HTTPS_PROXY",
  "NO_PROXY",
  "http_proxy",
  "https_proxy",
  "no_proxy",
  "SSL_CERT_FILE",
  "SSL_CERT_DIR",
  "REQUESTS_CA_BUNDLE",
  "CURL_CA_BUNDLE",
  "NODE_EXTRA_CA_CERTS",
  "PIP_CERT",
  "UV_NATIVE_TLS",
  "UV_INDEX_URL",
  "UV_EXTRA_INDEX_URL",
  "UV_PYTHON",
  "PLAYWRIGHT_BROWSERS_PATH",
  "TMUX_TMPDIR",
];

function checkNewRunId(runId) {
  if (
    typeof runId !== "string" ||
    !/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(runId)
  ) {
    throw new CliError(
      "--run-id must be letters, digits, '.', '_' or '-', starting with a letter or digit.",
      { code: 2 },
    );
  }
  const dir = join(verifyHome, runId);
  if (existsSync(dir)) {
    throw new CliError(`Run ${runId} already exists (${dir}).`, {
      code: 2,
      hint: "Pick a new --run-id, or reuse that run in place with `OH_VERIFY_RUN=<dir> control-openhands restart`.",
    });
  }
}

async function cmdLaunch({ flags }) {
  // A named run must be new: reusing its directory would overwrite the keys
  // and run.json of a run that may still be live. `restart` reuses a run.
  const requestedRunId = flags["run-id"];
  if (requestedRunId !== undefined) checkNewRunId(requestedRunId);
  if (nodeMajor() < 24) {
    throw new CliError(
      `Node >=24 is required by the launcher; this is ${process.version}.`,
      {
        code: 3,
        hint: "Put a Node 24 binary first on PATH (see package.json engines) and rerun.",
      },
    );
  }
  // One run (Agent Server, automation, frontend, Chromium) needs about
  // 1.5 GB. Several agents launching at once can exhaust a shared machine and
  // take every run down with it, so refuse instead of overcommitting.
  const availableMb = Math.round(availableMemoryBytes() / 1024 / 1024);
  const minimumMb = intFlag(flags["min-free-mb"], 2000);
  if (availableMb < minimumMb) {
    throw new CliError(
      `Only ${availableMb} MB of memory available; a run needs about 1500 MB.`,
      {
        code: 3,
        hint: "Stop runs you no longer need (`control-openhands runs`, `stop --run <dir>`), wait for other agents, or lower --min-free-mb deliberately.",
      },
    );
  }
  const uvx = spawnSync("uvx", ["--version"], { encoding: "utf8" });
  if (uvx.status !== 0) {
    throw new CliError(
      "uvx is not on PATH; the launcher runs Agent Server and automation through it.",
      {
        code: 3,
        hint: "Install uv (https://docs.astral.sh/uv/) and rerun.",
      },
    );
  }

  let existing;
  if (!flags.new) {
    try {
      existing = loadRun(flags, { required: false });
    } catch (error) {
      // Several live runs: reuse none of them, say how to start another.
      if (!(error instanceof CliError) || !/runs are alive/.test(error.message))
        throw error;
      throw new CliError(error.message, {
        code: error.code,
        hint: `${error.hint} To start another independent run, pass --new.`,
      });
    }
  }
  if (existing && groupAlive(existing.launcherPgid)) {
    out({
      ok: true,
      alreadyRunning: true,
      run: existing.dir,
      baseUrl: existing.baseUrl,
      hint: "Pass --new for a second, independent run.",
    });
    return;
  }

  // 1. Build the exact checkout (or reuse a build stamped with this revision).
  const revision = checkoutRevision();
  const currentBuildId = buildId();
  const buildDir = join(repoRoot, "build");
  const marker = join(buildDir, "verify-revision.txt");
  const buildMode = flags.build ?? "auto";
  const stamped = existsSync(marker) ? readFileSync(marker, "utf8").trim() : "";
  let built = false;
  if (
    buildMode === "always" ||
    (buildMode === "auto" && stamped !== currentBuildId)
  ) {
    if (!existsSync(join(repoRoot, "node_modules"))) {
      throw new CliError("node_modules is missing.", {
        code: 3,
        hint: "npm ci --ignore-scripts --no-audit --no-fund",
      });
    }
    process.stderr.write(
      `building ${revision} (build inputs ${currentBuildId}; npm run build:app)...\n`,
    );
    const result = spawnSync("npm", ["run", "build:app"], {
      cwd: repoRoot,
      stdio: ["ignore", 2, 2],
      env: { ...process.env, VITE_DO_NOT_TRACK: "1" },
    });
    if (result.status !== 0)
      throw new CliError("npm run build:app failed", { code: 3 });
    writeFileSync(marker, `${currentBuildId}\n`);
    built = true;
  } else if (
    buildMode === "never" &&
    !existsSync(join(buildDir, "index.html"))
  ) {
    throw new CliError("No build/ found and --build never was given.", {
      code: 3,
    });
  }

  // 2. Allocate a run directory and its keys.
  const runId =
    flags["run-id"] ||
    `${new Date().toISOString().replace(/[:.]/g, "").slice(0, 15)}-${randomBytes(3).toString("hex")}`;
  const dir = join(verifyHome, runId);
  const priv = join(dir, "private");
  mkdirSync(verifyHome, { recursive: true });
  try {
    mkdirSync(dir); // exclusive: fails if another launch took the id meanwhile
  } catch (error) {
    if (error.code === "EEXIST") checkNewRunId(runId);
    throw error;
  }
  for (const d of [
    priv,
    join(priv, "home"),
    join(priv, "state"),
    join(dir, "evidence"),
    join(dir, "workspace"),
  ]) {
    mkdirSync(d, { recursive: true, mode: d.startsWith(priv) ? 0o700 : 0o755 });
  }
  // MCP stdio servers start with a minimal environment (HOME, PATH), so a
  // CA bundle given to this process does not reach their uvx/npx. Seed the
  // private HOME's tool config with it.
  const caBundle = process.env.SSL_CERT_FILE || process.env.NODE_EXTRA_CA_CERTS;
  if (caBundle && existsSync(caBundle)) {
    mkdirSync(join(priv, "home", ".config", "uv"), { recursive: true });
    writeFileSync(
      join(priv, "home", ".config", "uv", "uv.toml"),
      "native-tls = true\n",
    );
    writeFileSync(join(priv, "home", ".npmrc"), `cafile=${caBundle}\n`);
  }
  writeFileSync(join(priv, "session-key"), randomBytes(32).toString("hex"), {
    mode: 0o600,
  });
  writeFileSync(join(priv, "encryption-key"), randomBytes(32).toString("hex"), {
    mode: 0o600,
  });

  // 3. The launcher's environment: only PASS_ENV, run-private paths and keys,
  // and the flags below reach it.
  const env = {};
  for (const name of PASS_ENV)
    if (process.env[name] !== undefined) env[name] = process.env[name];
  const realHome = process.env.HOME || "";
  Object.assign(env, {
    HOME: join(priv, "home"),
    // Reuse the operator's uv caches so each run does not re-download Python.
    UV_CACHE_DIR: process.env.UV_CACHE_DIR || join(realHome, ".cache", "uv"),
    UV_PYTHON_INSTALL_DIR:
      process.env.UV_PYTHON_INSTALL_DIR ||
      join(realHome, ".local", "share", "uv", "python"),
    OH_CANVAS_SAFE_STATE_DIR: join(priv, "state"),
    LOCAL_BACKEND_API_KEY: readFileSync(join(priv, "session-key"), "utf8"),
    OH_SECRET_KEY: readFileSync(join(priv, "encryption-key"), "utf8"),
    DO_NOT_TRACK: "1",
    VITE_DO_NOT_TRACK: "1",
    OPENHANDS_SUPPRESS_BANNER: "1",
  });
  if (flags["sdk-path"])
    env.OH_AGENT_SERVER_LOCAL_PATH = resolve(flags["sdk-path"]);
  if (flags["sdk-ref"]) env.OH_AGENT_SERVER_GIT_REF = flags["sdk-ref"];
  if (flags["automation-path"])
    env.OH_AUTOMATION_LOCAL_PATH = resolve(flags["automation-path"]);
  if (flags["automation-ref"])
    env.OH_AUTOMATION_GIT_REF = flags["automation-ref"];
  if (flags["sdk-version"]) env.OH_AGENT_SERVER_VERSION = flags["sdk-version"];
  if (flags["automation-version"])
    env.OH_AUTOMATION_VERSION = flags["automation-version"];
  if (flags.vscode) env.OH_CANVAS_ENABLE_VSCODE = "true";
  // Only PASS_ENV and the flags above reach the launcher, so a version pin
  // exported in the caller's shell would be dropped without a word.
  const ignoredEnv = Object.keys(process.env).filter(
    (k) =>
      (/^OH_(AGENT_SERVER|AUTOMATION)_/.test(k) ||
        k === "OH_CANVAS_ENABLE_VSCODE") &&
      !(k in env),
  );
  const warnings = [];
  if (ignoredEnv.length)
    warnings.push(
      `Ignored ${ignoredEnv.join(", ")} from your shell: runs are isolated. Use --sdk-version/--sdk-ref/--sdk-path, --automation-version/--automation-ref/--automation-path or --vscode.`,
    );

  // 4. Ports. The editor port is reserved only when this checkout's launcher
  // serves VS Code with this environment: opt-in since OpenHands/OpenHands#17660,
  // bundled before it and again since OpenHands/OpenHands#18048.
  const editor = editorOffset(repoRoot, env);
  if (editor.source === "fallback")
    warnings.push(
      `Could not ask scripts/dev-with-automation.mjs whether it serves VS Code (${editor.reason}); reserving the editor port as before.`,
    );
  const base = flags.port
    ? intFlag(flags.port)
    : await findPortBlock(
        intFlag(flags["port-from"], 18800),
        dir,
        editor.offset,
      );
  const ports = runPorts(base, editor.offset);
  if (flags.port) {
    const busy = [];
    for (const [name, port] of Object.entries(ports))
      if (await portOpen(port)) busy.push(`${port} (${name})`);
    if (busy.length)
      throw new CliError(`Ports busy: ${busy.join(", ")}`, {
        code: 3,
        hint: "Omit --port to pick a free block.",
      });
  }
  Object.assign(env, {
    OH_CANVAS_SAFE_BACKEND_PORT: String(ports.agentServer),
    OH_CANVAS_SAFE_AUTOMATION_PORT: String(ports.automation),
    OH_CANVAS_SAFE_VITE_PORT: String(ports.frontend),
    // A private tmux server per run (OpenHands/OpenHands#17946: the launcher's
    // <stateDir>/tmux is never created, so runs would share one server). It
    // lives under /tmp because a socket below the long run path would exceed
    // the 108-byte Unix socket limit.
    TMUX_TMPDIR: tmuxDirFor(dir),
  });
  const envWarning = warnings.length ? warnings.join(" ") : undefined;

  // 5. Start the repository's own production launcher, isolated.
  const launcherArgs = [
    join(repoRoot, "bin", "agent-canvas.mjs"),
    "--port",
    String(ports.ingress),
  ];
  if (flags.public) launcherArgs.push("--public");
  const savedEnv = { ...env };
  delete savedEnv.LOCAL_BACKEND_API_KEY;
  delete savedEnv.OH_SECRET_KEY;
  writeFileSync(join(priv, "launcher-env.json"), JSON.stringify(savedEnv), {
    mode: 0o600,
  });

  const run = {
    runId,
    dir,
    repo: repoRoot,
    revision,
    buildId: currentBuildId,
    builtThisRun: built,
    mode: flags.public ? "public" : "local",
    baseUrl: `http://127.0.0.1:${ports.ingress}`,
    ports,
    launcherArgs,
    startedAt: new Date().toISOString(),
    pins: defaults().versions,
    overrides: Object.fromEntries(
      Object.entries(env).filter(([k]) =>
        /^OH_(AGENT_SERVER_|AUTOMATION_|CANVAS_ENABLE_VSCODE$)/.test(k),
      ),
    ),
  };
  spawnLauncher(run);
  saveRun(run);
  try {
    if (
      existsSync(join(verifyHome, "current")) ||
      lstatSync(join(verifyHome, "current"), { throwIfNoEntry: false })
    ) {
      unlinkSync(join(verifyHome, "current"));
    }
  } catch {
    // no previous pointer
  }
  symlinkSync(dir, join(verifyHome, "current"));

  // 6. Wait for readiness: authenticated settings, automation health and SPA.
  await waitForStack(run, intFlag(flags.timeout, 600));
  const info = await http(run, "GET", "/server_info");
  run.versions = {
    agentServer: info.json?.version,
    sdk: info.json?.sdk_version ?? undefined,
  };
  run.readyAt = new Date().toISOString();
  saveRun(run);

  let browser;
  if (!flags["no-browser"]) {
    try {
      browser = await startBrowser(run);
    } catch (error) {
      if (!(error instanceof CliError)) throw error;
      // The stack is up and the run is saved: only the browser is missing.
      throw new CliError(error.message, {
        code: error.code,
        hint: `${error.hint} The stack is running: export OH_VERIFY_RUN=${dir}, fix the browser, then \`control-openhands browser start\` (or \`stop\` the run).`,
        extra: { ...(error.extra ?? {}), run: dir, baseUrl: run.baseUrl },
      });
    }
  }

  if (flags["print-run"]) {
    // For: export OH_VERIFY_RUN=$(control-openhands launch --new --print-run)
    if (envWarning) process.stderr.write(`warning: ${envWarning}\n`);
    process.stdout.write(`${dir}\n`);
    return;
  }
  out({
    ok: true,
    run: dir,
    ...(envWarning ? { warning: envWarning } : {}),
    baseUrl: run.baseUrl,
    mode: run.mode,
    revision,
    builtThisRun: built,
    ports,
    versions: run.versions,
    browser,
    next: [
      `export OH_VERIFY_RUN=${dir}`,
      "control-openhands doctor",
      run.mode === "public"
        ? "control-openhands login"
        : "control-openhands onboard --skip",
    ],
  });
}

async function waitForStack(run, timeoutSec) {
  const priv = join(run.dir, "private");
  const deadline = Date.now() + timeoutSec * 1000;
  let last = "";
  for (;;) {
    if (!groupAlive(run.launcherPgid)) {
      const tail = readFileSync(join(priv, "stack.log"), "utf8")
        .split("\n")
        .slice(-25)
        .join("\n");
      throw new CliError("The launcher exited before the stack became ready.", {
        code: 3,
        extra: { run: run.dir, logTail: tail.replace(/\x1b\[[0-9;]*m/g, "") },
      });
    }
    try {
      const [settings, automation, html] = await Promise.all([
        http(run, "GET", "/api/settings", { timeout: 4000 }),
        http(run, "GET", "/api/automation/health", { timeout: 4000 }),
        http(run, "GET", "/", { auth: false, timeout: 4000 }),
      ]);
      last = `settings=${settings.status} automation=${automation.status} ui=${html.status}`;
      if (
        settings.status === 200 &&
        automation.status === 200 &&
        html.status === 200
      )
        return;
    } catch (error) {
      last = String(error.message);
    }
    if (Date.now() > deadline) {
      throw new CliError(`Stack not ready after ${timeoutSec}s (${last}).`, {
        code: 3,
        hint: "Inspect <run>/private/stack.log; first launches download Python packages and can take minutes.",
        extra: { run: run.dir },
      });
    }
    await delay(1500);
  }
}

function spawnLauncher(run) {
  // The saved environment holds no secrets; keys are re-read from private/.
  const priv = join(run.dir, "private");
  const saved = JSON.parse(
    readFileSync(join(priv, "launcher-env.json"), "utf8"),
  );
  const env = launcherEnvFor(saved, process.env, PASS_ENV);
  env.LOCAL_BACKEND_API_KEY = readFileSync(join(priv, "session-key"), "utf8");
  env.OH_SECRET_KEY = readFileSync(join(priv, "encryption-key"), "utf8");
  if (saved.TMUX_TMPDIR) env.TMUX_TMPDIR = tmuxDirFor(run.dir);
  const logFd = openSync(join(priv, "stack.log"), "a", 0o600);
  const child = spawn(process.execPath, run.launcherArgs, {
    cwd: repoRoot,
    env,
    detached: true,
    stdio: ["ignore", logFd, logFd],
  });
  child.unref();
  run.launcherPid = child.pid;
  run.launcherPgid = child.pid;
  return child;
}

async function stopLauncher(run) {
  const pgid = run.launcherPgid;
  let forced = false;
  const serviceGroups = new Set(
    descendants(run.launcherPid)
      .map((p) => p.pgid)
      .filter((g) => g !== pgid),
  );
  if (groupAlive(pgid)) {
    const cmd = commandLine(pgid);
    if (cmd && !cmd.includes("agent-canvas.mjs")) {
      throw new CliError(
        `PID ${pgid} is no longer this run's launcher (${cmd.slice(0, 80)}); refusing to signal it.`,
        { code: 3 },
      );
    }
    process.kill(-pgid, "SIGTERM");
    for (let i = 0; i < 60 && groupAlive(pgid); i += 1) await delay(500);
    if (groupAlive(pgid)) {
      process.kill(-pgid, "SIGKILL");
      forced = true;
      await delay(1000);
    }
  }
  // The launcher shuts its services down itself; anything it left behind
  // (for example after a forced kill) is signalled by its own group.
  for (const group of serviceGroups) {
    if (!groupAlive(group)) continue;
    process.kill(-group, "SIGTERM");
    for (let i = 0; i < 20 && groupAlive(group); i += 1) await delay(500);
    if (groupAlive(group)) {
      process.kill(-group, "SIGKILL");
      forced = true;
    }
  }
  // Give sockets a moment to close before callers check or reuse the ports.
  for (const port of Object.values(run.ports)) {
    for (let i = 0; i < 20 && (await portOpen(port)); i += 1) await delay(500);
  }
  return { stopped: !groupAlive(pgid), forced };
}

async function cmdRestart({ flags }) {
  const run = loadRun(flags);
  if (
    !run.launcherArgs ||
    !existsSync(join(run.dir, "private", "launcher-env.json"))
  ) {
    throw new CliError("This run predates restart support; launch a new one.", {
      code: 3,
    });
  }
  // The checkout may have changed since launch (for example, VS Code bundled
  // again), so ask its launcher again which ports this run now uses. Asked
  // before stopping: an editor port the run did not use before cannot move
  // with the block, so a busy one refuses the restart while the run is still
  // up, as launch --port would.
  const saved = JSON.parse(
    readFileSync(join(run.dir, "private", "launcher-env.json"), "utf8"),
  );
  const next = portsAfterRestart(
    run.ports,
    editorOffset(repoRoot, launcherEnvFor(saved, process.env, PASS_ENV)),
  );
  if (next.added !== undefined && (await portOpen(next.added)))
    throw new CliError(
      `Port ${next.added} (vscode) is busy, and the checkout's launcher now serves VS Code there; restart refused, and the run is unchanged.`,
      {
        code: 3,
        hint: "The run is still up. Free that port and restart again, or launch --new for a free block.",
      },
    );
  const stopped = await stopLauncher(run);
  run.ports = next.ports;
  if (flags["rotate-key"]) {
    writeFileSync(
      join(run.dir, "private", "session-key"),
      randomBytes(32).toString("hex"),
      { mode: 0o600 },
    );
  }
  spawnLauncher(run);
  run.restartedAt = new Date().toISOString();
  saveRun(run);
  await waitForStack(run, intFlag(flags.timeout, 600));
  out({
    ok: true,
    run: run.dir,
    baseUrl: run.baseUrl,
    stopped,
    ports: run.ports,
    rotatedKey: Boolean(flags["rotate-key"]),
    ...(next.warning ? { warning: next.warning } : {}),
    note: flags["rotate-key"]
      ? "The browser still holds the old key: reload to reach the stale-key state (public mode shows the API-key prompt)."
      : "Same state, keys and port block; reload the browser to reconnect.",
  });
}

const SERVICES = {
  automation:
    /openhands-automation|automation[./](app|main|server)|uvicorn\S* .*automation/,
  "agent-server":
    /openhands-agent-server|openhands[./]agent_server|bin\/agent-server /,
  frontend: /static-server\.mjs/,
};

// The launcher starts each service in its own process group, so services are
// found by walking the process tree under the launcher, not by its group.
function descendants(rootPid) {
  const listing = spawnSync("ps", ["-eo", "pid=,ppid=,pgid=,args="], {
    encoding: "utf8",
  }).stdout;
  const rows = listing
    .split("\n")
    .map((line) => line.trim().match(/^(\d+)\s+(\d+)\s+(\d+)\s+(.*)$/))
    .filter(Boolean)
    .map((m) => ({ pid: +m[1], ppid: +m[2], pgid: +m[3], args: m[4] }));
  const found = [];
  const frontier = [rootPid];
  while (frontier.length) {
    const parent = frontier.pop();
    for (const row of rows) {
      if (row.ppid === parent) {
        found.push(row);
        frontier.push(row.pid);
      }
    }
  }
  return found;
}

async function cmdService({ positional, flags }) {
  const run = loadRun(flags);
  const [sub, name] = positional;
  const procs = descendants(run.launcherPid);
  if (sub === "status") {
    const services = {};
    for (const [service, pattern] of Object.entries(SERVICES)) {
      services[service] = procs
        .filter((p) => pattern.test(p.args))
        .map((p) => p.pid);
    }
    out({ ok: true, launcherAlive: groupAlive(run.launcherPgid), services });
    return;
  }
  if (sub === "stop") {
    const pattern = SERVICES[name];
    if (!pattern) {
      usage(
        `Unknown service ${name}. Known: ${Object.keys(SERVICES).join(", ")}`,
        "control-openhands service stop automation",
      );
    }
    // Only processes inside this run's own process group are candidates.
    const targets = procs.filter(
      (p) => pattern.test(p.args) && p.pid !== run.launcherPid,
    );
    if (!targets.length) {
      throw new CliError(
        `No ${name} process found in this run's process group.`,
        { hint: "control-openhands service status" },
      );
    }
    // Signal each service's own process group (uv and its Python child).
    for (const pgid of new Set(targets.map((t) => t.pgid))) {
      try {
        process.kill(-pgid, "SIGTERM");
      } catch {
        // already gone
      }
    }
    await delay(2000);
    out({
      ok: true,
      stopped: targets.map((t) => t.pid),
      launcherAlive: groupAlive(run.launcherPgid),
      next: "Drive the unavailable state, then `control-openhands restart` to bring the whole stack back.",
    });
    return;
  }
  usage(
    "Usage: control-openhands service status | service stop automation|agent-server|frontend",
    "control-openhands service stop automation",
  );
}

async function cmdStatus({ flags }) {
  const run = loadRun(flags);
  out({
    ok: true,
    run: run.dir,
    baseUrl: run.baseUrl,
    mode: run.mode,
    revision: run.revision,
    launcherAlive: groupAlive(run.launcherPgid),
    browserAlive: Boolean(daemonInfo(run)),
    ports: run.ports,
    versions: run.versions,
    startedAt: run.startedAt,
  });
}

async function cmdDoctor({ flags }) {
  const run = loadRun(flags);
  const checks = [];
  const add = (name, ok, detail, severity = "fail") =>
    checks.push({ name, ok, severity: ok ? "ok" : severity, detail });

  const alive = groupAlive(run.launcherPgid);
  const cmd = commandLine(run.launcherPid);
  add(
    "launcher process group alive",
    alive,
    alive ? cmd.slice(0, 120) : "launcher group is gone",
  );
  if (alive)
    add(
      "launcher is this run's agent-canvas",
      cmd.includes("agent-canvas.mjs"),
      cmd.slice(0, 120),
    );
  for (const [name, port] of Object.entries(run.ports)) {
    if (name === "vscode") continue;
    const open = await portOpen(port);
    add(`port ${name}:${port} listening`, open, open ? "open" : "closed");
  }
  if (alive) {
    try {
      const marker = await http(run, "GET", "/verify-revision.txt", {
        auth: false,
      });
      const served = marker.text.trim();
      add(
        "served build matches this checkout's build inputs",
        marker.status === 200 && served === run.buildId,
        `served=${served} run=${run.buildId} revision=${run.revision.slice(0, 12)}`,
      );
      const current = buildId();
      const unchanged = current === run.buildId;
      add(
        unchanged
          ? "checkout build inputs unchanged since launch"
          : "checkout build inputs changed since launch; relaunch with --build auto to serve them",
        unchanged,
        `launch=${run.buildId} now=${current}`,
        "warn",
      );
      const unauth = await http(run, "GET", "/api/settings", { auth: false });
      add(
        "unauthenticated API rejected",
        [401, 403].includes(unauth.status),
        `status=${unauth.status}`,
      );
      const settings = await http(run, "GET", "/api/settings");
      add(
        "authenticated settings readable",
        settings.status === 200 && Boolean(settings.json),
        `status=${settings.status}`,
      );
      const llm = settings.json?.agent_settings?.llm;
      const llmReady = Boolean(llm?.model && llm?.api_key);
      add(
        "LLM configured",
        llmReady,
        llmReady
          ? `model=${llm.model}`
          : `model=${llm?.model ?? "none"} without an API key: \`control-openhands llm preset deepseek\``,
        "info",
      );
      const info = await http(run, "GET", "/server_info");
      const pin = defaults().versions.agentServer;
      add(
        "agent-server reachable",
        info.status === 200,
        `version=${info.json?.version}`,
      );
      const version = info.json?.version;
      const overridden = Object.keys(run.overrides ?? {}).filter((k) =>
        k.startsWith("OH_AGENT_SERVER_"),
      );
      add(
        "agent-server matches pin",
        version === pin || overridden.length > 0,
        overridden.length
          ? `server=${version}, chosen by ${overridden.join(", ")} (default pin ${pin})`
          : `server=${version} pin=${pin}`,
        "warn",
      );
      const minimum = defaults().compatibility?.minimumAgentServer;
      if (minimum && version)
        add(
          "agent-server meets the UI minimum",
          versionAtLeast(version, minimum),
          versionAtLeast(version, minimum)
            ? `server=${version} minimum=${minimum}`
            : `server=${version} is below ${minimum}: every page shows the backend-compatibility gate instead of the app`,
          "warn",
        );
      const automation = await http(run, "GET", "/api/automation/health");
      add(
        "automation healthy",
        automation.status === 200,
        `status=${automation.status}`,
      );
    } catch (error) {
      add("HTTP checks", false, String(error.message));
    }
  }
  if (!flags["skip-ui"]) {
    if (daemonInfo(run)) {
      try {
        const probe = await browserCall(run, "uiprobe", {
          target: "/",
          timeout: 30_000,
        });
        const wedged =
          probe.markers.length === 1 && probe.markers[0] === "loading-spinner";
        add(
          "UI loads without page errors",
          probe.pageErrors.length === 0 && !wedged,
          `markers=${probe.markers.join(",") || "none"} pageErrors=${probe.pageErrors.length} testidsInDom=${probe.testids} (hidden included)`,
        );
      } catch (error) {
        add("UI probe", false, String(error.message));
      }
    } else {
      add(
        "browser daemon running",
        false,
        "start it with `control-openhands browser start`",
        "warn",
      );
    }
  }
  const failed = checks.filter((c) => c.severity === "fail");
  out({
    ok: failed.length === 0,
    run: run.dir,
    baseUrl: run.baseUrl,
    failed: failed.map((c) => `${c.name}: ${c.detail}`),
    warnings: checks
      .filter((c) => ["warn", "info"].includes(c.severity))
      .map((c) => `${c.name}: ${c.detail}`),
    passed: checks.filter((c) => c.ok).map((c) => `${c.name}: ${c.detail}`),
  });
  if (failed.length) process.exitCode = 3;
}

function tmuxDirFor(runDir) {
  const dir = tmuxPathFor(runDir);
  mkdirSync(dir, { recursive: true, mode: 0o700 });
  return dir;
}

async function cmdStop({ flags }) {
  const run = loadRun(flags);
  const result = { run: run.dir };
  result.browser = await stopBrowser(run);
  const pgid = run.launcherPgid;
  const launcher = await stopLauncher(run);
  if (launcher.forced) result.forced = true;
  const ports = {};
  for (const [name, port] of Object.entries(run.ports))
    ports[name] = (await portOpen(port)) ? "still open" : "closed";
  result.launcherStopped = !groupAlive(pgid);
  if (result.launcherStopped) {
    releaseClaim(run);
    rmSync(tmuxPathFor(run.dir), { recursive: true, force: true });
  }
  result.ports = ports;
  run.stoppedAt = new Date().toISOString();
  saveRun(run);
  const evidence = listFiles(join(run.dir, "evidence"));
  result.evidenceFiles = evidence.length;
  if (flags["purge-private"]) {
    if (!result.launcherStopped)
      throw new CliError(
        "Refusing to purge private state while processes are alive.",
        { code: 3 },
      );
    const privateDir = join(run.dir, "private");
    const real = realpathSync(privateDir);
    if (
      lstatSync(privateDir).isSymbolicLink() ||
      dirname(real) !== realpathSync(run.dir)
    ) {
      throw new CliError(
        "private/ is not a plain directory inside the run; not deleting.",
        { code: 3 },
      );
    }
    rmSync(real, { recursive: true, force: true });
    result.privatePurged = true;
    result.evidenceFilesAfterPurge = listFiles(
      join(run.dir, "evidence"),
    ).length;
  }
  result.ok =
    result.launcherStopped && Object.values(ports).every((v) => v === "closed");
  out(result);
  if (!result.ok) process.exitCode = 3;
}

function listFiles(dir) {
  if (!existsSync(dir)) return [];
  const files = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) files.push(...listFiles(path));
    else files.push(path);
  }
  return files;
}

async function cmdEnv({ flags }) {
  const run = loadRun(flags);
  process.stdout.write(
    `export OH_VERIFY_RUN=${run.dir}\nexport OH_VERIFY_BASE_URL=${run.baseUrl}\n`,
  );
}

async function cmdRuns() {
  const runs = [];
  if (existsSync(verifyHome)) {
    for (const name of readdirSync(verifyHome)) {
      const file = join(verifyHome, name, "run.json");
      if (name === "current" || !existsSync(file)) continue;
      const run = JSON.parse(readFileSync(file, "utf8"));
      runs.push({
        run: join(verifyHome, name),
        baseUrl: run.baseUrl,
        alive: groupAlive(run.launcherPgid),
        revision: run.revision,
        startedAt: run.startedAt,
      });
    }
  }
  out({ ok: true, home: verifyHome, runs });
}

// ---------------------------------------------------------------------------
// Commands: API and LLM arrangement.
// ---------------------------------------------------------------------------
async function cmdApi({ positional, flags }) {
  const run = loadRun(flags);
  const [method = "", path] = positional;
  const verb = method.toUpperCase();
  if (!path || !path.startsWith("/"))
    usage(
      "Usage: control-openhands api <METHOD> </path>",
      "control-openhands api GET /api/settings",
    );
  if (!["GET", "HEAD"].includes(verb) && !flags.write) {
    usage(
      `${verb} changes state. Pass --write to arrange a precondition; it is never UI proof.`,
      `control-openhands api ${verb} ${path} --write --data '{...}'`,
    );
  }
  let body;
  if (flags.data !== undefined) {
    const raw = String(flags.data).startsWith("@")
      ? readFileSync(String(flags.data).slice(1), "utf8")
      : String(flags.data);
    body = JSON.parse(raw);
  }
  // Non-JSON uploads (an automation tarball): --data-file F --content-type T.
  let rawBody;
  const headers = {};
  if (flags["data-file"] && flags["data-file"] !== true) {
    rawBody = readFileSync(resolve(String(flags["data-file"])));
    headers["Content-Type"] =
      flags["content-type"] && flags["content-type"] !== true
        ? String(flags["content-type"])
        : "application/octet-stream";
  }
  const response = await http(run, verb, path, {
    body,
    raw: rawBody,
    headers,
  });
  const max = intFlag(flags["max-bytes"], 6000);
  if (flags.pick && flags.pick !== true && response.json !== undefined) {
    // Paths start inside the response body; a leading "body." is accepted.
    const value = String(flags.pick)
      .replace(/^body\./, "")
      .split(".")
      .reduce((o, k) => (o == null ? undefined : o[k]), response.json);
    out({ ok: response.ok, status: response.status, pick: flags.pick, value });
    if (!response.ok) process.exitCode = 1;
    return;
  }
  out({
    ok: response.ok,
    status: response.status,
    warning:
      !["GET", "HEAD", "DELETE"].includes(verb) &&
      body === undefined &&
      rawBody === undefined
        ? `${verb} sent no body: pass --data '{...}' (JSON) or --data-file F --content-type T.`
        : undefined,
    contentType: response.contentType,
    body:
      response.json ??
      (response.text.length > max
        ? `${response.text.slice(0, max)}…`
        : response.text),
  });
  if (!response.ok) process.exitCode = 1;
}

function readApiKey(flags) {
  if (flags["api-key-file"])
    return readFileSync(resolve(String(flags["api-key-file"])), "utf8").trim();
  const envName =
    flags["api-key-env"] && flags["api-key-env"] !== true
      ? flags["api-key-env"]
      : undefined;
  if (envName) {
    const value = process.env[envName];
    if (!value)
      throw new CliError(`Environment variable ${envName} is empty.`, {
        code: 3,
      });
    return value.trim();
  }
  if (flags["api-key"]) {
    throw new CliError(
      "Do not pass keys on the command line (they land in shell history and logs).",
      { code: 2, hint: "Use --api-key-env NAME or --api-key-file PATH." },
    );
  }
  return undefined;
}

async function validateLlm(run, name, llm) {
  const result = await http(
    run,
    "POST",
    `/api/profiles/${encodeURIComponent(name)}/validate`,
    { body: { llm }, timeout: 120_000 },
  );
  if (!result.ok)
    throw new CliError(
      `Validation request failed: ${result.status} ${result.text.slice(0, 300)}`,
    );
  if (!result.json?.valid) {
    throw new CliError(`LLM settings for ${name} are invalid.`, {
      extra: { validation: result.json?.error },
    });
  }
  return result.json;
}

async function saveProfile(run, name, llm) {
  const saved = await http(
    run,
    "POST",
    `/api/profiles/${encodeURIComponent(name)}`,
    { body: { llm, include_secrets: true } },
  );
  if (!saved.ok)
    throw new CliError(
      `Saving LLM profile ${name} failed: ${saved.status} ${saved.text.slice(0, 300)}`,
    );
  return saved;
}

async function activateProfile(run, name) {
  const activated = await http(
    run,
    "POST",
    `/api/profiles/${encodeURIComponent(name)}/activate`,
    { body: {} },
  );
  if (!activated.ok)
    throw new CliError(
      `Activating LLM profile ${name} failed: ${activated.status} ${activated.text.slice(0, 300)}`,
    );
}

const PRESETS = {
  deepseek: {
    envVar: "DEEPSEEK_API_KEY",
    profiles: [
      {
        name: "deepseek-flash",
        model: "deepseek/deepseek-flash",
        activate: true,
      },
      { name: "deepseek-pro", model: "deepseek/deepseek-v4-pro" },
    ],
  },
};

async function cmdLlm({ positional, flags }) {
  const run = loadRun(flags);
  const [sub] = positional;
  if (sub === "show") {
    const settings = await http(run, "GET", "/api/settings");
    const profiles = await http(run, "GET", "/api/profiles");
    const llm = settings.json?.agent_settings?.llm ?? {};
    out({
      ok: true,
      active: {
        model: llm.model,
        base_url: llm.base_url,
        api_key_set: Boolean(llm.api_key),
      },
      profiles: profiles.json,
    });
    return;
  }
  if (sub === "set") {
    const name =
      flags.profile && flags.profile !== true ? flags.profile : undefined;
    const model = flags.model;
    if (!name || !model || model === true)
      usage(
        "llm set needs --profile and --model",
        "control-openhands llm set --profile deepseek-flash --model deepseek/deepseek-flash --api-key-env DEEPSEEK_API_KEY",
      );
    const apiKey = readApiKey(flags);
    if (!apiKey)
      usage("llm set needs --api-key-env NAME or --api-key-file PATH");
    const llm = { model, api_key: apiKey };
    if (flags["base-url"]) llm.base_url = flags["base-url"];
    if (!flags["no-validate"]) await validateLlm(run, name, llm);
    await saveProfile(run, name, llm);
    if (!flags["no-activate"]) await activateProfile(run, name);
    const settings = await http(run, "GET", "/api/settings");
    out({
      ok: true,
      profile: name,
      model,
      activated: !flags["no-activate"],
      activeModel: settings.json?.agent_settings?.llm?.model,
    });
    return;
  }
  if (sub === "preset") {
    const presetName = positional[1];
    const preset = PRESETS[presetName];
    if (!preset)
      usage(
        `Unknown preset ${presetName}. Known: ${Object.keys(PRESETS).join(", ")}`,
        "control-openhands llm preset deepseek",
      );
    const apiKey = readApiKey({
      ...flags,
      "api-key-env": flags["api-key-env"] ?? preset.envVar,
    });
    const created = [];
    for (const profile of preset.profiles) {
      const llm = { model: profile.model, api_key: apiKey };
      if (!flags["no-validate"]) await validateLlm(run, profile.name, llm);
      await saveProfile(run, profile.name, llm);
      created.push(profile.name);
    }
    const active = preset.profiles.find((p) => p.activate);
    if (active) await activateProfile(run, active.name);
    const settings = await http(run, "GET", "/api/settings");
    out({
      ok: true,
      preset: presetName,
      profiles: created,
      active: active?.name,
      activeModel: settings.json?.agent_settings?.llm?.model,
    });
    return;
  }
  if (sub === "check") {
    // One 1-token completion through the agent-server's own pre-flight
    // endpoint, without saving anything.
    const model = flags.model;
    const apiKey = readApiKey(flags);
    if (!model || model === true || !apiKey)
      usage(
        "llm check needs --model and --api-key-env/--api-key-file",
        "control-openhands llm check --model deepseek/deepseek-flash --api-key-env DEEPSEEK_API_KEY",
      );
    const llm = { model, api_key: apiKey };
    if (flags["base-url"]) llm.base_url = flags["base-url"];
    out({ ok: true, model, ...(await validateLlm(run, "check", llm)) });
    return;
  }
  usage(
    "Usage: control-openhands llm show|set|preset|check",
    "control-openhands llm preset deepseek",
  );
}

// ---------------------------------------------------------------------------
// Commands: essential user pathways (driven through the real UI).
// ---------------------------------------------------------------------------
async function visible(run, selector) {
  const { count } = await browserCall(run, "count", { selector });
  if (!count) return false;
  return (await browserCall(run, "visible", { selector })).visible;
}

async function cmdLogin({ flags }) {
  const run = loadRun(flags);
  if (run.mode !== "public") {
    out({
      ok: true,
      skipped: true,
      reason: "local mode injects the session key; no login screen",
    });
    return;
  }
  // Public mode asks for the key in one of two places: the first-run
  // onboarding's "Add a backend" step, or the API-key screen shown once
  // onboarding is done or skipped. The key is typed, never printed.
  await browserCall(run, "goto", { target: "/" });
  for (let i = 0; i < 20; i += 1) {
    if (await visible(run, "testid=onboarding-backend-api-key")) {
      await browserCall(run, "fill", {
        selector: "testid=onboarding-backend-api-key",
        value: sessionKey(run),
      });
      await browserCall(run, "click", {
        selector: "testid=onboarding-backend-next",
      });
      await browserCall(run, "wait", {
        selector: "testid=onboarding-step-choose-agent",
        timeout: 30_000,
      });
      out({
        ok: true,
        loggedIn: true,
        via: "onboarding backend step",
        next: "control-openhands onboard (continues at Choose your agent)",
      });
      return;
    }
    if (await visible(run, "testid=api-key-entry-screen")) {
      // Connect stays disabled until Host Name is filled.
      const name = await browserCall(run, "value", {
        selector: "testid=api-key-entry-name",
      }).catch(() => ({ value: "x" }));
      if (!name.value) {
        await browserCall(run, "fill", {
          selector: "testid=api-key-entry-name",
          value: "Local",
        });
      }
      await browserCall(run, "fill", {
        selector: "testid=api-key-entry-api-key",
        value: sessionKey(run),
      });
      await browserCall(run, "click", {
        selector: "testid=api-key-entry-submit",
      });
      await browserCall(run, "wait", {
        selector: "testid=api-key-entry-screen",
        state: "hidden",
        timeout: 30_000,
      });
      out({ ok: true, loggedIn: true, via: "api-key-entry-screen" });
      return;
    }
    if (await visible(run, "testid=root-layout")) {
      out({ ok: true, loggedIn: true, via: "already authenticated" });
      return;
    }
    await delay(500);
  }
  throw new CliError(
    "Neither the onboarding backend step nor the API-key screen appeared.",
    {
      hint: "Inspect with `control-openhands browser testids` and `browser snapshot`.",
    },
  );
}

async function cmdOnboard({ flags }) {
  const run = loadRun(flags);
  const steps = [];
  await browserCall(run, "goto", { target: "/" });
  await delay(1500);
  if (await visible(run, "testid=telemetry-consent-form")) {
    const analytics = flags.analytics === "on";
    const box = "testid=telemetry-consent-form >> role=checkbox";
    const checked = (
      await browserCall(run, "eval", {
        expression:
          "document.querySelector('[data-testid=telemetry-consent-form] input[type=checkbox]')?.checked ?? null",
      })
    ).value;
    if (checked !== null && checked !== analytics)
      await browserCall(run, "click", { selector: box });
    await browserCall(run, "click", {
      selector: "testid=confirm-telemetry-preferences",
    });
    steps.push(`telemetry consent: analytics ${analytics ? "on" : "off"}`);
    await delay(500);
  }
  if (await visible(run, "testid=onboarding-modal")) {
    if (flags.skip || flags.skip === "") {
      await browserCall(run, "click", { selector: "testid=onboarding-skip" });
      steps.push("onboarding: skipped (not an onboarding-success proof)");
    } else {
      const agent =
        flags.agent && flags.agent !== true ? flags.agent : "openhands";
      await browserCall(run, "click", {
        selector: `testid=onboarding-agent-option-${agent}`,
      });
      await browserCall(run, "click", {
        selector: "testid=onboarding-agent-next",
      });
      steps.push(`onboarding: chose ${agent}`);
      await browserCall(run, "wait", {
        selector: "testid=onboarding-step-setup-llm",
      });
      await browserCall(run, "click", {
        selector: "testid=onboarding-llm-next",
      });
      steps.push("onboarding: LLM step continued with the current settings");
      await browserCall(run, "wait", {
        selector: "testid=onboarding-step-say-hello",
      });
      await browserCall(run, "click", {
        selector: "testid=onboarding-hello-close",
      });
      steps.push("onboarding: closed at say-hello");
    }
    await browserCall(run, "wait", {
      selector: "testid=onboarding-modal",
      state: "hidden",
      timeout: 15_000,
    });
  }
  if (!steps.length)
    steps.push("nothing to do: no consent form or onboarding modal shown");
  out({ ok: true, steps, url: (await browserCall(run, "url")).url });
}

const TERMINAL = [
  "idle",
  "finished",
  "error",
  "stuck",
  "paused",
  "waiting_for_confirmation",
];

async function conversationInfo(run, id) {
  const res = await http(
    run,
    "GET",
    `/api/conversations/${encodeURIComponent(id)}`,
  );
  if (!res.ok) throw new CliError(`Conversation ${id}: HTTP ${res.status}`);
  return res.json;
}

function summarize(info) {
  return {
    id: info.id,
    title: info.title,
    status: info.execution_status,
    model: info.current_model_id ?? info.agent?.llm?.model,
    workspace: info.workspace?.working_dir,
    cost:
      info.metrics?.accumulated_cost ??
      info.stats?.usage_to_metrics?.agent?.accumulated_cost,
    updatedAt: info.updated_at,
  };
}

async function waitConversation(run, id, { until, timeoutSec, fresh }) {
  const wanted = until === "any" || !until ? TERMINAL : until.split(",");
  const deadline = Date.now() + timeoutSec * 1000;
  let info = await conversationInfo(run, id);
  // --fresh: a message was just sent; ignore the previous run's terminal
  // status until the conversation has run again.
  let sawRunning = !fresh && info.execution_status === "running";
  const startedAt = Date.now();
  while (Date.now() < deadline) {
    info = await conversationInfo(run, id);
    if (info.execution_status === "running") sawRunning = true;
    const terminal = wanted.includes(info.execution_status);
    // A fresh conversation reports idle before its first step: wait for it
    // to actually run unless the caller only asked for "idle".
    const settled = fresh
      ? sawRunning || Date.now() - startedAt > 20_000
      : sawRunning || until === "idle" || info.execution_status !== "idle";
    if (terminal && settled) {
      // Titles are generated shortly after the first run finishes; report
      // the status that matched, not a later one.
      const matched = summarize(info);
      for (
        let i = 0;
        i < 5 && !matched.title && Date.now() < deadline;
        i += 1
      ) {
        await delay(2000);
        matched.title = (await conversationInfo(run, id)).title ?? null;
      }
      return { ...matched, waitedFor: wanted };
    }
    await delay(2000);
  }
  throw new CliError(
    `Conversation ${id} still ${info.execution_status} after ${timeoutSec}s`,
    { extra: summarize(info) },
  );
}

function eventText(event) {
  const pick = (v) =>
    typeof v === "string"
      ? v
      : Array.isArray(v)
        ? v.map((p) => p.text ?? "").join(" ")
        : "";
  return (
    pick(event.llm_message?.content) ||
    pick(event.message) ||
    event.action?.command ||
    event.action?.message ||
    event.action?.path ||
    pick(event.observation?.content) ||
    event.observation?.text ||
    [event.code, event.detail].filter(Boolean).join(": ") ||
    ""
  ).replace(/\s+/g, " ");
}

// The Open Workspace folder browser has no path field: walk it to PATH.
async function openWorkspace(run, name, { stay } = {}) {
  // A bare name resolves inside <run>/workspace, where fixtures live.
  const inRun = join(run.dir, "workspace", name);
  const path = !existsSync(name) && existsSync(inRun) ? inRun : name;
  if (!existsSync(path))
    throw new CliError(`No such folder: ${path}`, {
      code: 2,
      hint: "Create it first, for example: control-openhands fixture git-repo --name qa-repo",
    });
  const want = realpathSync(resolve(path));
  if (!stay) await browserCall(run, "goto", { target: "/" });
  await browserCall(run, "click", {
    selector: "testid=open-workspace-button",
    timeout: 30_000,
  });
  await browserCall(run, "click", { selector: "testid=workspace-dropdown" });
  await browserCall(run, "click", { selector: "testid=add-workspaces-button" });
  const walked = await browserCall(run, "pick-folder", {
    path: want,
    timeout: 30_000,
  });
  await browserCall(run, "click", { selector: "testid=folder-browser-use" });
  await browserCall(run, "click", {
    selector: "testid=workspace-launch-button",
    timeout: 30_000,
  });
  return { workspace: want, steps: walked.steps };
}

async function cmdWorkspace({ positional, flags }) {
  const run = loadRun(flags);
  const [sub, path] = positional;
  if (sub !== "open" || !path)
    usage(
      "Usage: control-openhands workspace open PATH [--stay]",
      "control-openhands workspace open qa-repo   # a name resolves in <run>/workspace",
    );
  out({ ok: true, ...(await openWorkspace(run, path, { stay: flags.stay })) });
}

async function cmdConversation({ positional, flags }) {
  const run = loadRun(flags);
  const [sub, id] = positional;
  const timeoutSec = intFlag(flags.timeout, 300);
  if (sub === "start") {
    const prompt = flags.prompt;
    if (!prompt || prompt === true)
      usage(
        "conversation start needs --prompt",
        'control-openhands conversation start --prompt "Create hello.py that prints hi, run it" --wait',
      );
    let picked;
    if (flags.workspace && flags.workspace !== true)
      picked = await openWorkspace(run, String(flags.workspace), {
        stay: flags.stay,
      });
    else if (!flags["stay"]) await browserCall(run, "goto", { target: "/" });
    // Home and conversation pages share the composer: a contenteditable
    // testid=chat-input with testid=submit-button beside it.
    await browserCall(run, "wait", {
      selector: "testid=chat-input",
      timeout: 30_000,
    });
    await browserCall(run, "fill", {
      selector: "testid=chat-input",
      value: prompt,
    });
    await browserCall(run, "click", { selector: "testid=submit-button" });
    const { url } = await browserCall(run, "wait-url", {
      pattern: "/conversations/[^/?#]+",
      timeout: 60_000,
    });
    const conversationId = /\/conversations\/([^/?#]+)/.exec(url)[1];
    const result = {
      ok: true,
      id: conversationId,
      url,
      workspace: picked?.workspace,
    };
    if (flags.wait)
      Object.assign(
        result,
        await waitConversation(run, conversationId, {
          until: flags.until,
          timeoutSec,
        }),
      );
    out(result);
    return;
  }
  if (sub === "wait") {
    if (!id)
      usage(
        "conversation wait <id>",
        "control-openhands conversation wait <id> --until finished,idle --timeout 300",
      );
    out({
      ok: true,
      ...(await waitConversation(run, id, {
        until: flags.until,
        timeoutSec,
        fresh: Boolean(flags.fresh),
      })),
    });
    return;
  }
  if (sub === "status") {
    if (!id) usage("conversation status <id>");
    out({ ok: true, ...summarize(await conversationInfo(run, id)) });
    return;
  }
  if (sub === "list") {
    const res = await http(run, "GET", "/api/conversations/search?limit=50");
    const items = res.json?.items ?? res.json ?? [];
    out({
      ok: res.ok,
      count: items.length,
      conversations: items.map(summarize),
    });
    return;
  }
  if (sub === "events") {
    if (!id) usage("conversation events <id> [--last N]");
    const last = intFlag(flags.last, 25);
    const res = await http(
      run,
      "GET",
      `/api/conversations/${encodeURIComponent(id)}/events/search?limit=100&sort_order=${flags["from-start"] ? "TIMESTAMP" : "TIMESTAMP_DESC"}`,
    );
    if (!res.ok)
      throw new CliError(
        `events: HTTP ${res.status} ${res.text.slice(0, 200)}`,
      );
    const fetched = res.json?.items ?? [];
    const items = flags["from-start"]
      ? fetched.slice(0, last)
      : fetched.slice(0, last).reverse();
    const grep =
      flags.grep && flags.grep !== true
        ? String(flags.grep).toLowerCase()
        : null;
    const kinds =
      flags.kinds && flags.kinds !== true
        ? String(flags.kinds).split(",")
        : null;
    const shown = items
      .filter((e) => !kinds || kinds.includes(e.kind))
      .map((e) => {
        const text = eventText(e);
        const max = flags.full ? 4000 : 160;
        const row = {
          kind: e.kind,
          source: e.source,
          tool: e.tool_name ?? e.action?.kind,
          text: text.slice(0, max),
          truncated: text.length > max || undefined,
          skills: e.activated_skills?.length ? e.activated_skills : undefined,
          tools: Array.isArray(e.tools)
            ? e.tools.map((tool) => tool.title || tool.name || tool.kind)
            : undefined,
          ts: e.timestamp,
        };
        if (grep) {
          // Search the whole event (system prompt, tool args, extended
          // content), not only the summarized text.
          const raw = JSON.stringify(e);
          const lower = raw.toLowerCase();
          const excerpts = [];
          let at = lower.indexOf(grep);
          let matches = 0;
          while (at >= 0) {
            matches += 1;
            if (excerpts.length < 3)
              excerpts.push(
                raw.slice(Math.max(0, at - 60), at + grep.length + 60),
              );
            at = lower.indexOf(grep, at + grep.length);
          }
          row.matches = matches;
          row.excerpts = excerpts.length ? excerpts : undefined;
        }
        return row;
      })
      .filter((row) => !grep || row.matches > 0);
    out({
      ok: true,
      total: fetched.length,
      count: shown.length,
      events: shown,
    });
    return;
  }
  if (sub === "pause") {
    // Arrange step (not UI proof): pause through the API, as the status
    // menu's Stop Runtime does, to reach paused-only states.
    if (!id) usage("conversation pause <id>");
    const before = await conversationInfo(run, id);
    // Local Stop Runtime interrupts (cancels the in-flight LLM call).
    const res = await http(
      run,
      "POST",
      `/api/conversations/${encodeURIComponent(id)}/interrupt`,
    );
    if (!res.ok)
      throw new CliError(
        `interrupt: HTTP ${res.status} ${res.text.slice(0, 200)}`,
      );
    if (before.execution_status !== "running") {
      out({
        ok: true,
        ...summarize(await conversationInfo(run, id)),
        note: `It was ${before.execution_status}: pause only changes a running conversation.`,
      });
      return;
    }
    out({
      ok: true,
      ...(await waitConversation(run, id, {
        until: "paused,idle,finished,error",
        timeoutSec: 90,
      })),
    });
    return;
  }
  if (sub === "send") {
    // A follow-up message through the conversation page's composer.
    const prompt = flags.prompt;
    if (!id || !prompt || prompt === true)
      usage(
        "conversation send <id> --prompt TEXT [--wait]",
        'control-openhands conversation send <id> --prompt "Now add a test" --wait',
      );
    await browserCall(run, "goto", {
      target: `/conversations/${encodeURIComponent(id)}`,
    });
    await browserCall(run, "wait", {
      selector: "testid=chat-input",
      timeout: 30_000,
    });
    await browserCall(run, "fill", {
      selector: "testid=chat-input",
      value: String(prompt),
    });
    await browserCall(run, "click", { selector: "testid=submit-button" });
    const result = { ok: true, id, sent: String(prompt).length };
    if (flags.wait)
      Object.assign(
        result,
        await waitConversation(run, id, {
          until: flags.until,
          timeoutSec,
          fresh: true,
        }),
      );
    out(result);
    return;
  }
  usage(
    "Usage: control-openhands conversation start|send|wait|status|list|events|pause",
    'control-openhands conversation start --prompt "..." --wait',
  );
}

// ---------------------------------------------------------------------------
// Commands: fixtures (disposable, inside the run directory).
// ---------------------------------------------------------------------------
function crc32(buf) {
  let c;
  const table = [];
  for (let n = 0; n < 256; n += 1) {
    c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  let crc = 0xffffffff;
  for (const byte of buf) crc = table[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function pngBytes(width, height, [r, g, b]) {
  const chunk = (type, data) => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length);
    const body = Buffer.concat([Buffer.from(type), data]);
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(body));
    return Buffer.concat([len, body, crc]);
  };
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header[8] = 8;
  header[9] = 2;
  const rows = [];
  for (let y = 0; y < height; y += 1) {
    const row = Buffer.alloc(1 + width * 3);
    for (let x = 0; x < width; x += 1) {
      const stripe = Math.floor(x / (width / 4)) % 2 === 0;
      row[1 + x * 3] = stripe ? r : 255 - r;
      row[2 + x * 3] = stripe ? g : 255 - g;
      row[3 + x * 3] = stripe ? b : 255 - b;
    }
    rows.push(row);
  }
  return Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    chunk("IHDR", header),
    chunk("IDAT", deflateSync(Buffer.concat(rows))),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

const MCP_FIXTURE_SOURCE = `#!/usr/bin/env node
// QA fixture: minimal MCP server over stdio (JSON-RPC 2.0, one tool).
import { appendFileSync } from "node:fs";
import { createInterface } from "node:readline";
const log = new URL("./requests.log", import.meta.url);
const send = (msg) => process.stdout.write(JSON.stringify(msg) + "\\n");
createInterface({ input: process.stdin }).on("line", (line) => {
  let req;
  try { req = JSON.parse(line); } catch { return; }
  const { id, method, params } = req;
  try { appendFileSync(log, new Date().toISOString() + " " + method + "\\n"); } catch {}
  if (id === undefined) return; // notifications
  if (method === "resources/list") return send({ jsonrpc: "2.0", id, result: { resources: [] } });
  if (method === "resources/templates/list") return send({ jsonrpc: "2.0", id, result: { resourceTemplates: [] } });
  if (method === "prompts/list") return send({ jsonrpc: "2.0", id, result: { prompts: [] } });
  if (method === "initialize")
    return send({ jsonrpc: "2.0", id, result: {
      protocolVersion: params?.protocolVersion ?? "2025-06-18",
      capabilities: { tools: {} },
      serverInfo: { name: "qa-mcp", version: "1.0.0" } } });
  if (method === "tools/list")
    return send({ jsonrpc: "2.0", id, result: { tools: [{
      name: "qa_echo",
      description: "Echo the given text back, prefixed with QA_ECHO:",
      inputSchema: { type: "object", properties: { text: { type: "string" } }, required: ["text"] } }] } });
  if (method === "tools/call")
    return send({ jsonrpc: "2.0", id, result: { content: [{ type: "text",
      text: "QA_ECHO: " + String(params?.arguments?.text ?? "") }] } });
  if (method === "ping") return send({ jsonrpc: "2.0", id, result: {} });
  send({ jsonrpc: "2.0", id, error: { code: -32601, message: "Method not found: " + method } });
});
`;

async function cmdFixture({ positional, flags }) {
  const run = loadRun(flags);
  const [kind] = positional;
  const workspace = join(run.dir, "workspace");
  const name =
    flags.name && flags.name !== true
      ? String(flags.name).replace(/[^\w.-]+/g, "_")
      : undefined;
  if (kind === "git-repo") {
    const dir = join(workspace, name ?? "qa-repo");
    if (existsSync(join(dir, ".git"))) {
      out({ ok: true, path: dir, existed: true });
      return;
    }
    mkdirSync(join(dir, "src"), { recursive: true });
    writeFileSync(
      join(dir, "README.md"),
      "# QA fixture repository\n\nDisposable repository created by control-openhands.\n",
    );
    writeFileSync(
      join(dir, "src", "calc.py"),
      "def add(a, b):\n    return a + b\n",
    );
    writeFileSync(
      join(dir, "src", "test_calc.py"),
      "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    );
    const gitEnv = {
      ...process.env,
      GIT_AUTHOR_NAME: "QA Fixture",
      GIT_AUTHOR_EMAIL: "qa@example.invalid",
      GIT_COMMITTER_NAME: "QA Fixture",
      GIT_COMMITTER_EMAIL: "qa@example.invalid",
    };
    for (const args of [
      ["init", "-q", "-b", "main"],
      ["add", "."],
      ["commit", "-q", "-m", "Initial fixture commit"],
    ]) {
      execFileSync("git", args, { cwd: dir, env: gitEnv });
    }
    const remote =
      flags.remote && flags.remote !== true ? String(flags.remote) : undefined;
    // A remote URL lets the UI show repo/branch links and Pull/Push chips;
    // nothing is fetched or pushed.
    if (remote)
      execFileSync("git", ["remote", "add", "origin", remote], { cwd: dir });
    out({
      ok: true,
      path: dir,
      files: ["README.md", "src/calc.py", "src/test_calc.py"],
      branch: "main",
      remote,
    });
    return;
  }
  if (kind === "image") {
    const dir = join(run.dir, "evidence", "_fixtures");
    mkdirSync(dir, { recursive: true });
    const path = join(dir, `${name ?? "qa-image"}.png`);
    writeFileSync(
      path,
      pngBytes(
        intFlag(flags.width, 160),
        intFlag(flags.height, 96),
        [32, 120, 220],
      ),
    );
    out({ ok: true, path, bytes: readFileSync(path).length });
    return;
  }
  if (kind === "file") {
    const dir = join(run.dir, "evidence", "_fixtures");
    mkdirSync(dir, { recursive: true });
    const path = join(dir, name ?? "qa-note.txt");
    writeFileSync(
      path,
      flags.content && flags.content !== true
        ? String(flags.content)
        : "QA fixture file created by control-openhands.\n",
    );
    out({ ok: true, path });
    return;
  }
  if (kind === "tarball") {
    // A one-file script bundle for the automation "Upload tarball" action:
    // main.py prints pong, packed as <name>.tar.gz next to its source folder.
    const dir = join(run.dir, "evidence", "_fixtures");
    const base = name ?? "qa-tarball";
    const src = join(dir, base);
    mkdirSync(src, { recursive: true });
    writeFileSync(join(src, "main.py"), 'print("pong")\n');
    const path = join(dir, `${base}.tar.gz`);
    execFileSync("tar", ["-czf", path, "-C", src, "main.py"]);
    out({ ok: true, path, entrypoint: "python3 main.py", files: ["main.py"] });
    return;
  }
  if (kind === "mcp-server") {
    // A dependency-free stdio MCP server with one tool, qa_echo, for the
    // custom-server and agent-use recipes (no network, no npm install).
    const dir = join(workspace, name ?? "qa-mcp");
    mkdirSync(dir, { recursive: true });
    const path = join(dir, "server.mjs");
    writeFileSync(path, MCP_FIXTURE_SOURCE);
    out({
      ok: true,
      path,
      command: process.execPath,
      args: [path],
      tool: "qa_echo",
      hint: `Add a custom STDIO server with command ${process.execPath} and argument ${path}; ask the agent to call qa_echo.`,
    });
    return;
  }
  if (kind === "git-remote") {
    // A bare repository to push to (Git Sync, Push): a local "remote".
    const dir = join(workspace, `${name ?? "qa-remote"}.git`);
    if (!existsSync(dir))
      execFileSync("git", ["init", "-q", "--bare", "-b", "main", dir]);
    out({
      ok: true,
      path: dir,
      url: `file://${dir}`,
      hint: "Inspect what arrived with: git -C <path> log --oneline --all",
    });
    return;
  }
  if (kind === "skill") {
    // A personal skill under the run's private HOME, or a project skill in a
    // fixture repo (--repo NAME).
    const skillName = name ?? "qa-hello";
    const root =
      flags.repo && flags.repo !== true
        ? join(workspace, String(flags.repo))
        : join(run.dir, "private", "home");
    if (flags.repo && !existsSync(root))
      usage(
        `No fixture repo ${root}`,
        "control-openhands fixture git-repo --name qa-repo",
      );
    const dir = join(root, ".agents", "skills", skillName);
    mkdirSync(dir, { recursive: true });
    const trigger =
      flags.trigger && flags.trigger !== true
        ? String(flags.trigger)
        : undefined;
    const body =
      flags.body && flags.body !== true
        ? String(flags.body)
        : "When this skill is active, end your reply with the word QA_SKILL_OK.";
    const path = join(dir, "SKILL.md");
    writeFileSync(
      path,
      `---\nname: ${skillName}\ndescription: QA fixture skill created by control-openhands.\n${trigger ? `triggers:\n- ${trigger}\n` : ""}---\n\n${body}\n`,
    );
    out({
      ok: true,
      path,
      scope: flags.repo ? "project" : "user",
      hint: "Reload the page: the skills list is cached.",
    });
    return;
  }
  if (kind === "folder") {
    const dir = join(workspace, name ?? "qa-folder");
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "notes.md"), "# QA folder fixture\n");
    out({ ok: true, path: dir });
    return;
  }
  usage(
    "Usage: control-openhands fixture git-repo|git-remote|folder|image|file|tarball|skill|mcp-server [--name N] [--remote URL]",
    "control-openhands fixture git-repo --name qa-repo",
  );
}

// ---------------------------------------------------------------------------
// Commands: browser.
// ---------------------------------------------------------------------------
// Central-directory listing of a .zip download (names only).
function zipEntries(buf, limit = 30) {
  const eocd = buf.lastIndexOf(Buffer.from([0x50, 0x4b, 0x05, 0x06]));
  if (eocd < 0 || eocd + 22 > buf.length) return undefined;
  const count = buf.readUInt16LE(eocd + 10);
  let off = buf.readUInt32LE(eocd + 16);
  const names = [];
  for (let i = 0; i < count && off + 46 <= buf.length; i += 1) {
    if (buf.readUInt32LE(off) !== 0x02014b50) break;
    const nameLen = buf.readUInt16LE(off + 28);
    const extraLen = buf.readUInt16LE(off + 30);
    const commentLen = buf.readUInt16LE(off + 32);
    names.push(buf.toString("utf8", off + 46, off + 46 + nameLen));
    off += 46 + nameLen + extraLen + commentLen;
  }
  return { count, names: names.slice(0, limit) };
}

// Member names of a .tar or .tar.gz download (ustar headers).
function tarEntries(buf, limit = 30) {
  let data = buf;
  if (buf.length > 2 && buf[0] === 0x1f && buf[1] === 0x8b) {
    try {
      data = gunzipSync(buf);
    } catch {
      return undefined;
    }
  }
  if (data.length < 512 || data.toString("latin1", 257, 262) !== "ustar")
    return undefined;
  const names = [];
  let count = 0;
  for (let off = 0; off + 512 <= data.length; ) {
    const name = data.toString("utf8", off, off + 100).replace(/\0.*$/s, "");
    if (!name) break;
    const prefix = data
      .toString("utf8", off + 345, off + 500)
      .replace(/\0.*$/s, "");
    const size = parseInt(
      data
        .toString("latin1", off + 124, off + 136)
        .replace(/\0.*$/s, "")
        .trim() || "0",
      8,
    );
    count += 1;
    if (names.length < limit) names.push(prefix ? `${prefix}/${name}` : name);
    off += 512 + Math.ceil(size / 512) * 512;
  }
  return { gzip: data !== buf, count, names };
}

function inspectDownload(path, needle) {
  const buf = readFileSync(path);
  const info = { bytes: buf.length };
  const isZip = buf.length >= 4 && buf.readUInt32LE(0) === 0x04034b50;
  const tar = isZip ? undefined : tarEntries(buf);
  if (isZip) info.zip = zipEntries(buf);
  else if (tar) info.tar = tar;
  else if (!buf.subarray(0, 4096).includes(0))
    info.head = buf.toString("utf8", 0, 600);
  if (needle !== undefined)
    info.contains = isZip
      ? (zipEntries(buf, Infinity)?.names ?? []).some((n) => n.includes(needle))
      : tar
        ? (tarEntries(buf, Infinity)?.names ?? []).some((n) =>
            n.includes(needle),
          )
        : buf.toString("utf8").includes(needle);
  return info;
}

async function cmdBrowser({ positional, flags }) {
  const run = loadRun(flags);
  const [verb, ...rest] = positional;
  const timeout =
    flags.timeout !== undefined ? intFlag(flags.timeout) : undefined;
  const sel = rest[0];
  const need = (n, example) => {
    if (rest.length < n)
      usage(`browser ${verb} needs ${n} argument(s)`, example);
  };
  let result;
  switch (verb) {
    case "start":
      result = await startBrowser(run);
      break;
    case "stop":
      result = await stopBrowser(run);
      break;
    case "goto":
    case "open":
      result = await browserCall(run, "goto", {
        target: rest[0] ?? "/",
        allowExternal: Boolean(flags["allow-external"]),
      });
      break;
    case "reload":
    case "back":
    case "forward":
    case "url":
    case "tabs":
      result = await browserCall(run, verb);
      break;
    case "downloads": {
      result = await browserCall(run, verb);
      const last = intFlag(flags.last, 0);
      if (last) result.downloads = result.downloads.slice(-last);
      if (flags.inspect || flags.contains) {
        const needle =
          flags.contains && flags.contains !== true
            ? String(flags.contains)
            : undefined;
        for (const d of result.downloads)
          if (d.path && existsSync(d.path))
            d.inspect = inspectDownload(d.path, needle);
      }
      break;
    }
    case "click":
    case "dblclick": {
      need(1, `control-openhands browser ${verb} 'testid=submit-button'`);
      const position =
        flags.position && flags.position !== true
          ? (([x, y]) => ({ x: Number(x), y: Number(y) }))(
              String(flags.position).split(","),
            )
          : undefined;
      result = await browserCall(run, verb, {
        selector: sel,
        timeout,
        force: Boolean(flags.force),
        button: flags.button,
        modifiers:
          flags.modifiers && flags.modifiers !== true
            ? String(flags.modifiers).split(",")
            : undefined,
        position,
        expectUrl: flags["expect-url"],
        expectNewUrl: flags["expect-new-url"],
        hoverFirst: flags["hover-first"],
        observe: flags.observe,
        observeMs: flags["observe-ms"],
      });
      break;
    }
    case "mouse-click":
      need(2, "control-openhands browser mouse-click 20 400");
      result = await browserCall(run, "mouse-click", {
        x: rest[0],
        y: rest[1],
        button: flags.button,
      });
      break;
    case "tooltip":
      need(
        1,
        "control-openhands browser tooltip 'testid=chat-dictation-button'",
      );
      result = await browserCall(run, "tooltip", { selector: sel, timeout });
      break;
    case "media":
      result = await browserCall(run, "media", { clear: Boolean(flags.clear) });
      break;
    case "toasts":
      result = await browserCall(run, "toasts", {
        history: Boolean(flags.history),
        clear: Boolean(flags.clear),
      });
      break;
    case "clipboard":
      result = await browserCall(run, "clipboard", {
        write:
          flags.write !== undefined && flags.write !== true
            ? String(flags.write)
            : undefined,
      });
      break;
    case "drag":
      need(
        1,
        "control-openhands browser drag 'testid=card-a' 'testid=card-b' [--position before|after] | drag 'testid=divider' --by -200,0",
      );
      if (!rest[1] && !flags.by)
        usage("browser drag needs a target selector or --by DX,DY");
      result = await browserCall(run, "drag", {
        source: sel,
        target: rest[1],
        by: flags.by,
        steps: flags.steps,
        position: flags.position,
      });
      break;
    case "upload-via":
      need(
        2,
        "control-openhands browser upload-via 'role=menuitem[name=\"Add Files and Images\"]' ./qa.png",
      );
      result = await browserCall(run, "upload-via", {
        trigger: sel,
        files: rest.slice(1).map((f) => resolve(f)),
        timeout,
      });
      break;
    case "drop-files":
      need(
        2,
        "control-openhands browser drop-files 'testid=chat-input' ./qa.png",
      );
      result = await browserCall(run, "drop-files", {
        selector: sel,
        files: rest.slice(1).map((f) => resolve(f)),
        stage: flags.stage,
      });
      break;
    case "paste": {
      need(
        1,
        "control-openhands browser paste 'testid=chat-input' --file ./qa.png",
      );
      const files = [].concat(flags.file ?? []).filter((f) => f !== true);
      result = await browserCall(run, "paste", {
        selector: sel,
        text:
          flags.text !== undefined && flags.text !== true
            ? String(flags.text)
            : undefined,
        files: files.map((f) => resolve(String(f))),
      });
      break;
    }
    case "choose":
      need(
        2,
        "control-openhands browser choose 'label=Model' 'deepseek/deepseek-flash'",
      );
      result = await browserCall(run, "choose", {
        selector: sel,
        option: rest[1],
        timeout,
      });
      break;
    case "reset": {
      // A fresh browser profile: first-run state, empty localStorage, no
      // cached backend entries. The stack keeps running.
      const stopped = await stopBrowser(run);
      rmSync(join(run.dir, "private", "browser-profile"), {
        recursive: true,
        force: true,
      });
      result = { stopped, profileCleared: true, ...(await startBrowser(run)) };
      break;
    }
    case "network":
      result = await browserCall(run, "network", {
        clear: Boolean(flags.clear),
        external: Boolean(flags.external),
        last: flags.last,
        filter:
          flags.filter && flags.filter !== true ? flags.filter : undefined,
      });
      break;
    case "hover":
    case "focus":
    case "check":
    case "uncheck":
    case "count":
    case "visible":
    case "enabled":
    case "value":
    case "text":
    case "bbox":
      need(1, `control-openhands browser ${verb} 'testid=submit-button'`);
      result = await browserCall(run, verb, {
        selector: sel,
        timeout,
        force: Boolean(flags.force),
        button: flags.button,
        reveal: Boolean(flags.reveal),
      });
      break;
    case "fill":
    case "type": {
      need(
        1,
        `control-openhands browser ${verb} 'testid=name-input' 'QA value'`,
      );
      let value = rest[1];
      if (flags["value-file"])
        value = readFileSync(resolve(flags["value-file"]), "utf8");
      if (flags["value-env"]) value = process.env[flags["value-env"]] ?? "";
      if (value === undefined)
        usage(`browser ${verb} needs a value (or --value-file / --value-env)`);
      result = await browserCall(run, verb, { selector: sel, value, timeout });
      break;
    }
    case "press":
      need(1, "control-openhands browser press Escape [--selector 'testid=x']");
      result = await browserCall(run, "press", {
        key: rest[0],
        selector: flags.selector,
        timeout,
      });
      break;
    case "select":
      need(2, "control-openhands browser select 'testid=x' value");
      result = await browserCall(run, "select", {
        selector: sel,
        value: rest[1],
        timeout,
      });
      break;
    case "upload":
      need(2, "control-openhands browser upload 'input[type=file]' ./file.png");
      result = await browserCall(run, "upload", {
        selector: sel,
        files: rest.slice(1).map((f) => resolve(f)),
        timeout,
      });
      break;
    case "scroll":
      result = await browserCall(run, "scroll", {
        selector: sel,
        by: flags.by,
        x: flags.x,
        timeout,
      });
      break;
    case "wait":
      need(
        1,
        "control-openhands browser wait 'testid=x' [--state hidden] [--timeout 30000]",
      );
      result = await browserCall(run, "wait", {
        selector: sel,
        state: flags.state,
        timeout,
      });
      break;
    case "wait-url":
      need(1, "control-openhands browser wait-url '/settings/llm$'");
      result = await browserCall(run, "wait-url", {
        pattern: rest[0],
        timeout,
      });
      break;
    case "wait-text":
      need(1, "control-openhands browser wait-text 'Saved'");
      result = await browserCall(run, "wait-text", { text: rest[0], timeout });
      break;
    case "attr":
      need(2, "control-openhands browser attr 'testid=x' aria-expanded");
      result = await browserCall(run, "attr", {
        selector: sel,
        name: rest[1],
        timeout,
      });
      break;
    case "snapshot":
      result = await browserCall(run, "snapshot", {
        selector: sel,
        maxLines: flags["max-lines"],
        feature: flags.feature,
        name: flags.name,
      });
      break;
    case "testids":
      result = await browserCall(run, "testids", {
        selector: sel,
        includeHidden: Boolean(flags.hidden),
      });
      if (flags.filter)
        result.testids = result.testids.filter((t) =>
          t.testid.includes(flags.filter),
        );
      result.count = result.testids.length;
      break;
    case "screenshot":
      if (!flags.feature || !flags.name)
        usage(
          "browser screenshot needs --feature <ID> and --name <label>",
          "control-openhands browser screenshot --feature F14.create --name after-reload",
        );
      result = await browserCall(run, "screenshot", {
        feature: flags.feature,
        name: flags.name,
        selector: sel,
        fullPage: Boolean(flags["full-page"]),
      });
      break;
    case "clock":
      if (
        !flags["offset-ms"] &&
        flags.system === undefined &&
        flags.fixed === undefined
      )
        usage(
          "browser clock needs --offset-ms N | --system ISO|+MS | --fixed ISO|+MS",
          "control-openhands browser clock --offset-ms 300000   # five minutes ahead",
        );
      result = await browserCall(run, "clock", {
        offsetMs: flags["offset-ms"],
        system: flags.system,
        fixed: flags.fixed,
      });
      break;
    case "viewport":
      need(
        1,
        "control-openhands browser viewport phone   # desktop|phone|narrow|tablet|WxH",
      );
      result = await browserCall(run, "viewport", { size: rest[0] });
      break;
    case "errors":
      result = await browserCall(run, "errors", {
        clear: Boolean(flags.clear),
        all: Boolean(flags.all),
        appOnly: Boolean(flags["app-only"]),
        noWarnings: Boolean(flags["no-warnings"]),
      });
      break;
    case "wait-tab":
      need(
        1,
        "control-openhands browser wait-tab 'auth\\.openai\\.com' [--timeout 10000]",
      );
      result = await browserCall(run, "wait-tab", {
        pattern: rest[0],
        timeout,
      });
      break;
    case "events":
      result = await browserCall(run, "events", {
        kinds: flags.kinds,
        last: flags.last,
      });
      break;
    case "eval":
      need(1, 'control-openhands browser eval "document.title"');
      result = await browserCall(run, "eval", { expression: rest[0] });
      break;
    case "tab":
      need(1, "control-openhands browser tab 1");
      result = await browserCall(run, "tab", { index: rest[0] });
      break;
    case "close-tab":
      need(1, "control-openhands browser close-tab 1");
      result = await browserCall(run, "close-tab", { index: rest[0] });
      break;
    case "dialogs":
      result = await browserCall(run, "dialogs", { policy: flags.policy });
      break;
    case "storage":
      result = await browserCall(run, "storage", {
        keysOnly: !flags.values,
        session: Boolean(flags.session),
      });
      break;
    default:
      usage(
        `Unknown browser verb ${verb ?? ""}`,
        "control-openhands browser --help",
      );
  }
  out({ ok: true, ...result });
}

// ---------------------------------------------------------------------------
// Commands: evidence ledger.
// ---------------------------------------------------------------------------
const RESULTS = ["pass", "fail", "blocked", "not-run"];

// The ledger is append-only: the report shows each check's latest row,
// keyed by feature and entry point, minus retractions.
function latestRows(rows) {
  const latest = new Map();
  for (const row of rows) {
    const key = `${row.feature}|${row.entry ?? ""}`;
    if (row.retracted) {
      // A retraction without --entry drops every row of that feature.
      for (const k of [...latest.keys()]) {
        if (k === key || (!row.entry && k.startsWith(`${row.feature}|`)))
          latest.delete(k);
      }
    } else latest.set(key, row);
  }
  return latest;
}

// What changed between two ledgers' latest rows (same feature and entry).
function compareLedgers(before, after) {
  const newlyFailing = [];
  const newlyPassing = [];
  const other = [];
  const missing = [];
  const entry = (row) => ({
    feature: row.feature,
    entry: row.entry,
    note: row.note,
  });
  for (const [key, row] of after) {
    const prev = before.get(key);
    if (prev?.result === row.result) continue;
    const change = { ...entry(row), before: prev?.result, after: row.result };
    if (row.result === "fail" && prev?.result !== "fail")
      newlyFailing.push(change);
    else if (row.result === "pass" && prev && prev.result !== "pass")
      newlyPassing.push(change);
    else other.push(change);
  }
  for (const [key, row] of before)
    if (!after.has(key))
      missing.push({ ...entry(row), before: row.result, after: undefined });
  return { newlyFailing, newlyPassing, other, missing };
}

async function cmdEvidence({ positional, flags }) {
  const run = loadRun(flags);
  const ledger = join(run.dir, "evidence", "ledger.jsonl");
  const [sub] = positional;
  if (sub === "add") {
    if (!flags.feature || !RESULTS.includes(flags.result)) {
      usage(
        "evidence add needs --feature ID and --result pass|fail|blocked|not-run",
        "control-openhands evidence add --feature F14.create --result pass --entry 'Settings > Secrets > Add' --expected 'row after reload' --actual 'row present' --artifact evidence/F14.create/after-reload.png",
      );
    }
    const artifacts = []
      .concat(flags.artifact ?? [])
      .filter((a) => a !== true)
      .flatMap((a) => String(a).split(","))
      .filter(Boolean);
    const missing = artifacts.filter((a) => !existsSync(resolve(run.dir, a)));
    if (missing.length) {
      throw new CliError(`Artifact not found: ${missing.join(", ")}`, {
        code: 2,
        hint: "Paths are relative to the run directory (for example evidence/F14.list/list.png) or absolute.",
      });
    }
    const page = daemonInfo(run)
      ? await browserCall(run, "url").catch(() => ({}))
      : {};
    const row = {
      ts: new Date().toISOString(),
      feature: flags.feature,
      entry: flags.entry === true ? undefined : flags.entry,
      result: flags.result,
      expected: flags.expected === true ? undefined : flags.expected,
      actual: flags.actual === true ? undefined : flags.actual,
      artifacts,
      note: flags.note === true ? undefined : flags.note,
      revision: run.revision,
      backend: `${run.mode} agent-server ${run.versions?.agentServer ?? "?"}`,
      url: page.url,
      viewport: page.viewport
        ? `${page.viewport.width}x${page.viewport.height}`
        : undefined,
    };
    appendFileSync(ledger, `${JSON.stringify(row)}\n`);
    const known = new Set(mapIdList().map((i) => i.id));
    const warning =
      known.has(row.feature) || /^BUG-/.test(row.feature)
        ? undefined
        : `${row.feature} is not a sub-feature ID in the map. Use an ID from the family's "## Sub-features" list (add it there first), or BUG-<id> for bug repros.`;
    out({ ok: true, recorded: row, warning });
    return;
  }
  if (sub === "retract") {
    if (!flags.feature)
      usage("evidence retract needs --feature ID [--entry E] [--note why]");
    const row = {
      ts: new Date().toISOString(),
      feature: flags.feature,
      entry: flags.entry === true ? undefined : flags.entry,
      retracted: true,
      note: flags.note === true ? undefined : flags.note,
    };
    appendFileSync(ledger, `${JSON.stringify(row)}\n`);
    out({ ok: true, retracted: row });
    return;
  }
  const rows = existsSync(ledger)
    ? readFileSync(ledger, "utf8")
        .trim()
        .split("\n")
        .filter(Boolean)
        .map((l) => JSON.parse(l))
    : [];
  if (sub === "list") {
    const filtered = flags.feature
      ? rows.filter((r) => r.feature.startsWith(flags.feature))
      : rows;
    out({ ok: true, count: filtered.length, rows: filtered });
    return;
  }
  if (sub === "report") {
    const latest = latestRows(rows);
    const counts = Object.fromEntries(RESULTS.map((r) => [r, 0]));
    for (const row of latest.values()) counts[row.result] += 1;
    // Per family: the denominator the report contract asks for.
    const families = new Map();
    for (const row of latest.values()) {
      const id = row.feature.slice(0, 3);
      if (!families.has(id))
        families.set(id, Object.fromEntries(RESULTS.map((r) => [r, 0])));
      families.get(id)[row.result] += 1;
    }
    // --baseline PATH: another run's ledger.jsonl (or run directory). Rows
    // are compared by feature and entry point; what changed is listed first.
    let changes;
    if (flags.baseline && flags.baseline !== true) {
      let path = resolve(String(flags.baseline));
      if (existsSync(join(path, "evidence", "ledger.jsonl")))
        path = join(path, "evidence", "ledger.jsonl");
      if (!existsSync(path))
        throw new CliError(`No ledger at ${path}`, { code: 2 });
      const before = latestRows(
        readFileSync(path, "utf8")
          .trim()
          .split("\n")
          .filter(Boolean)
          .map((l) => JSON.parse(l)),
      );
      changes = compareLedgers(before, latest);
    }
    if (flags.json) {
      out({
        ok: true,
        run: run.runId,
        revision: run.revision,
        checks: latest.size,
        counts,
        families: Object.fromEntries(
          [...families.entries()].sort(([a], [b]) => a.localeCompare(b)),
        ),
        rows: [...latest.values()],
        ...(changes ? { changes } : {}),
      });
      return;
    }
    const cell = (v) =>
      String(v ?? "")
        .replace(/\|/g, "\\|")
        .replace(/\n/g, " ");
    const lines = [
      `# Verification ledger — run ${run.runId}`,
      "",
      `Revision \`${run.revision}\`, ${run.mode} mode, Agent Server ${run.versions?.agentServer ?? "?"}, base ${run.baseUrl}.`,
      "",
      `Checks: ${latest.size} — ${RESULTS.map((r) => `${r} ${counts[r]}`).join(", ")}. Families: ${families.size}.`,
      "",
    ];
    if (changes) {
      lines.push(`## Changes since baseline`, "");
      const section = (title, list) => {
        if (!list.length) return;
        lines.push(`${title}:`, "");
        for (const c of list)
          lines.push(
            `- ${c.feature}${c.entry ? ` (${c.entry})` : ""}: ${c.before ?? "absent"} → ${c.after ?? "absent"}${c.note ? ` — ${c.note}` : ""}`,
          );
        lines.push("");
      };
      section("Newly failing", changes.newlyFailing);
      section("Newly passing", changes.newlyPassing);
      section("Other changes", changes.other);
      section("Not checked this run", changes.missing);
      if (
        !changes.newlyFailing.length &&
        !changes.newlyPassing.length &&
        !changes.other.length &&
        !changes.missing.length
      )
        lines.push("No result changed against the baseline.", "");
    }
    lines.push(
      "## Families",
      "",
      "| Family | pass | fail | blocked | not-run |",
      "|---|---|---|---|---|",
    );
    for (const [id, c] of [...families.entries()].sort(([a], [b]) =>
      a.localeCompare(b),
    ))
      lines.push(
        `| ${id} | ${c.pass} | ${c.fail} | ${c.blocked} | ${c["not-run"]} |`,
      );
    lines.push(
      "",
      "## Checks",
      "",
      "| Feature/check | Entry point | Expected → actual | Result | Note | Evidence |",
      "|---|---|---|---|---|---|",
    );
    // Fail and blocked rows first (report.md), then by feature ID.
    const order = { fail: 0, blocked: 1, "not-run": 2, pass: 3 };
    for (const row of [...latest.values()].sort(
      (a, b) =>
        order[a.result] - order[b.result] || a.feature.localeCompare(b.feature),
    )) {
      lines.push(
        `| ${cell(row.feature)} | ${cell(row.entry)} | ${cell(row.expected)} → ${cell(row.actual)} | ${row.result} | ${cell(row.note)} | ${row.artifacts.map((a) => `\`${cell(relative(run.dir, resolve(run.dir, a)))}\``).join(" ")} |`,
      );
    }
    process.stdout.write(`${lines.join("\n")}\n`);
    return;
  }
  usage(
    "Usage: control-openhands evidence add|list|report",
    "control-openhands evidence report > report.md",
  );
}

// ---------------------------------------------------------------------------
// Commands: feature-map tooling (index hygiene and coverage).
// ---------------------------------------------------------------------------
const REQUIRED_H2 = [
  "Sub-features",
  "How to get to it (user POV)",
  "Driving it with control-openhands",
  "Gotchas",
];
const KNOWN_COMMANDS = new Set([
  "restart",
  "service",
  "launch",
  "status",
  "doctor",
  "stop",
  "env",
  "runs",
  "api",
  "llm",
  "login",
  "onboard",
  "conversation",
  "fixture",
  "workspace",
  "browser",
  "evidence",
  "map",
  "help",
]);
const BROWSER_VERBS = new Set([
  "start",
  "stop",
  "goto",
  "open",
  "reload",
  "back",
  "forward",
  "url",
  "tabs",
  "tab",
  "close-tab",
  "downloads",
  "click",
  "dblclick",
  "hover",
  "focus",
  "check",
  "uncheck",
  "count",
  "visible",
  "enabled",
  "value",
  "text",
  "bbox",
  "fill",
  "type",
  "press",
  "select",
  "upload",
  "scroll",
  "wait",
  "wait-url",
  "wait-text",
  "attr",
  "snapshot",
  "testids",
  "screenshot",
  "viewport",
  "clock",
  "errors",
  "events",
  "eval",
  "dialogs",
  "storage",
  "mouse-click",
  "tooltip",
  "media",
  "network",
  "toasts",
  "reset",
  "clipboard",
  "drag",
  "upload-via",
  "drop-files",
  "paste",
  "choose",
  "wait-tab",
]);

function featureFiles() {
  if (!existsSync(mapDir)) return [];
  return readdirSync(mapDir)
    .filter((f) => f.endsWith(".md") && f !== "README.md")
    .sort();
}

function mapCheck({ only } = {}) {
  const problems = [];
  const ids = new Map();
  const index = existsSync(join(mapDir, "README.md"))
    ? readFileSync(join(mapDir, "README.md"), "utf8")
    : "";
  if (!index) problems.push("references/feature-map/README.md is missing");
  // The baseline line, when present, must be the only one (a merge can leave
  // two) and carry a full SHA and a date: map affected and the passes read it.
  const baselines = baselineLines(index);
  if (baselines.length > 1)
    problems.push(
      `README.md: ${baselines.length} Maintenance baseline lines; keep the one the latest pass proposed`,
    );
  for (const line of baselines) {
    if (!parseBaseline(line))
      problems.push(
        "README.md: the Maintenance baseline line must read `Maintenance baseline: main@<sha> (<YYYY-MM-DD>)`",
      );
    else if (!/main@[0-9a-f]{40} /.test(line))
      problems.push(
        "README.md: the Maintenance baseline line must carry the full 40-character SHA",
      );
  }
  const files = featureFiles();
  for (const file of files) {
    const text = readFileSync(join(mapDir, file), "utf8");
    // --file checks one entry while the index is being written by someone else.
    if (!only && !index.includes(`(${file})`) && !index.includes(`(./${file})`))
      problems.push(`${file}: not linked from README.md`);
    if (!/^# .+/m.test(text.split("\n")[0]))
      problems.push(`${file}: must start with an H1 title`);
    const h2 = [...text.matchAll(/^## (.+)$/gm)].map((m) => m[1].trim());
    if (JSON.stringify(h2) !== JSON.stringify(REQUIRED_H2))
      problems.push(
        `${file}: H2 sections must be exactly ${REQUIRED_H2.join(" / ")} (found ${h2.join(" / ")})`,
      );
    const sub =
      text.split(/^## /m).find((s) => s.startsWith("Sub-features")) ?? "";
    const declared = [...sub.matchAll(/`(F\d{2}\.[a-z0-9-]+)`/g)].map(
      (m) => m[1],
    );
    if (!declared.length)
      problems.push(`${file}: no sub-feature IDs like \`F01.example\``);
    for (const id of declared) {
      if (ids.has(id))
        problems.push(`${file}: duplicate ID ${id} (also in ${ids.get(id)})`);
      ids.set(id, file);
    }
    const drive =
      text.split(/^## /m).find((s) => s.startsWith("Driving it")) ?? "";
    if (!/^Preconditions:/m.test(drive))
      problems.push(
        `${file}: Driving section must start with "Preconditions:"`,
      );
    for (const m of text.matchAll(
      /control-openhands ([a-z-]+)(?: ([a-z-]+))?/g,
    )) {
      if (!KNOWN_COMMANDS.has(m[1]))
        problems.push(`${file}: unknown command \`control-openhands ${m[1]}\``);
      if (m[1] === "browser" && m[2] && !BROWSER_VERBS.has(m[2]))
        problems.push(`${file}: unknown browser verb \`${m[2]}\``);
    }
    for (const m of text.matchAll(/\]\((\.\.\/[^)#]+)/g)) {
      const target = resolve(mapDir, m[1]);
      if (!existsSync(target)) problems.push(`${file}: dead link ${m[1]}`);
    }
    // The Source: line is what drift checks and `map affected` read: every
    // path it names must exist (node_modules content is not checked out).
    const head = familyHead(text);
    if (!head.sources.length)
      problems.push(`${file}: no Source: line with implementation paths`);
    for (const s of head.sources) {
      if (s.path.startsWith("node_modules/")) continue;
      if (!sourcePathExists(s))
        problems.push(`${file}: Source: path ${s.path} does not exist`);
    }
  }
  // References (`Fnn.slug` in prose, --feature Fnn.slug in recipes) must name
  // a declared ID once that family's file exists.
  const families = new Set(
    files.map((f) => /^(F\d{2})-/.exec(f)?.[1]).filter(Boolean),
  );
  for (const file of files) {
    if (only && file !== only) continue;
    const text = readFileSync(join(mapDir, file), "utf8");
    const refs = new Set();
    for (const m of text.matchAll(/`(F\d{2}\.[a-z0-9-]+)`/g)) refs.add(m[1]);
    for (const m of text.matchAll(/--feature (F\d{2}\.[a-z0-9-]+)/g))
      refs.add(m[1]);
    for (const ref of refs) {
      if (!ids.has(ref) && families.has(ref.slice(0, 3)))
        problems.push(`${file}: reference to unknown ID ${ref}`);
    }
  }
  for (const m of index.matchAll(/\]\(\.?\/?([A-Za-z0-9-]+\.md)\)/g)) {
    if (!existsSync(join(mapDir, m[1])))
      problems.push(`README.md: links missing file ${m[1]}`);
  }
  // The index's counts go stale whenever a family gains an ID; a whole-map
  // check compares them with the files.
  if (!only) {
    const counts = new Map();
    for (const file of ids.values())
      counts.set(file, (counts.get(file) ?? 0) + 1);
    for (const row of index.matchAll(
      /^\|\s*F\d{2}\s*\|\s*\[[^\]]*\]\(\.?\/?([A-Za-z0-9-]+\.md)\).*\|\s*(\d+)\s*\|\s*$/gm,
    )) {
      const actual = counts.get(row[1]) ?? 0;
      if (Number(row[2]) !== actual)
        problems.push(
          `README.md: ${row[1]} lists ${row[2]} sub-features, the file declares ${actual}`,
        );
    }
    const total = /(\d+) families, (\d+) sub-features/.exec(index);
    if (
      total &&
      (Number(total[1]) !== files.length || Number(total[2]) !== ids.size)
    )
      problems.push(
        `README.md: says "${total[0]}", the map has ${files.length} families and ${ids.size} sub-features`,
      );
  }
  return {
    files: files.length,
    ids: ids.size,
    checked: only ?? "all",
    problems,
  };
}

// Rewrites the README index's per-family sub-feature counts and its total
// from the files; everything else in the index is left as written.
function fixIndexCounts() {
  const path = join(mapDir, "README.md");
  if (!existsSync(path)) return;
  const counts = new Map();
  for (const { file } of mapIdList())
    counts.set(file, (counts.get(file) ?? 0) + 1);
  let total = 0;
  for (const n of counts.values()) total += n;
  const text = readFileSync(path, "utf8")
    .replace(
      /^(\|\s*F\d{2}\s*\|\s*\[[^\]]*\]\(\.?\/?([A-Za-z0-9-]+\.md)\).*\|\s*)(\d+)(\s*\|\s*)$/gm,
      (row, head, file, n, tail) =>
        counts.has(file) ? `${head}${counts.get(file)}${tail}` : row,
    )
    .replace(
      /(\d+) families, (\d+) sub-features/,
      `${featureFiles().length} families, ${total} sub-features`,
    );
  writeFileSync(path, text);
}

// A Source: path spec exists when the file or directory is there, or, for a
// `name-*.tsx` glob, when at least one file in its directory matches.
function sourcePathExists(s) {
  const full = join(repoRoot, s.path);
  if (!s.glob) return existsSync(full);
  const dir = dirname(full);
  if (!existsSync(dir)) return false;
  const [head, tail = ""] = basename(s.path).split("*");
  return readdirSync(dir).some(
    (name) => name.startsWith(head) && name.endsWith(tail),
  );
}

// Families with their parsed Source: lines.
function familyList() {
  return featureFiles().map((file) => {
    const text = readFileSync(join(mapDir, file), "utf8");
    return {
      id: /^(F\d{2})/.exec(file)?.[1] ?? file,
      file,
      ...familyHead(text),
    };
  });
}

function routePaths() {
  const source = readFileSync(join(repoRoot, "src", "routes.ts"), "utf8");
  const routes = [];
  // route("settings", "routes/settings.tsx", [ ...children ]) nests by brackets.
  const parents = [];
  let lastRoutePath;
  const tokens = source.matchAll(
    /(route|index)\(\s*(?:"([^"]*)"\s*,\s*)?"(routes\/[^"]+)"|\[|\]/g,
  );
  for (const t of tokens) {
    if (t[0] === "[") {
      parents.push(lastRoutePath);
      lastRoutePath = undefined;
      continue;
    }
    if (t[0] === "]") {
      parents.pop();
      continue;
    }
    const prefix = parents.filter(Boolean).join("/");
    const path =
      t[1] === "index"
        ? `/${prefix}`
        : `/${[prefix, t[2]].filter(Boolean).join("/")}`;
    routes.push({ path: path.replace(/\/+/g, "/"), file: `src/${t[3]}` });
    lastRoutePath = t[1] === "route" ? t[2] : undefined;
  }
  return routes;
}

function mapCoverage() {
  const corpus = featureFiles()
    .map((f) => readFileSync(join(mapDir, f), "utf8"))
    .join("\n");
  // The index's "## Not mapped" section lists deliberate exclusions with a
  // reason; they are reported as excluded, not as gaps.
  const index = existsSync(join(mapDir, "README.md"))
    ? readFileSync(join(mapDir, "README.md"), "utf8")
    : "";
  const notMapped = (index.split(/^## Not mapped/m)[1] ?? "").split(/^## /m)[0];
  const routes = routePaths().map((r) => {
    const pattern = routePattern(r.path);
    const byPath = pattern.test(corpus);
    return {
      ...r,
      byPath,
      mapped: byPath || corpus.includes(r.file),
      excluded: pattern.test(notMapped),
    };
  });
  const featureDirs = existsSync(
    join(repoRoot, "src", "components", "features"),
  )
    ? readdirSync(join(repoRoot, "src", "components", "features"), {
        withFileTypes: true,
      })
        .filter((d) => d.isDirectory())
        .map((d) => d.name)
    : [];
  const components = featureDirs.map((name) => ({
    dir: `src/components/features/${name}`,
    mapped: corpus.includes(`components/features/${name}`),
    excluded: notMapped.includes(`components/features/${name}`),
  }));
  return {
    routes: {
      total: routes.length,
      unmapped: routes.filter((r) => !r.mapped && !r.excluded),
      excluded: routes.filter((r) => !r.mapped && r.excluded),
      // Counted as mapped only because their route module is cited: worth
      // documenting the user-facing path too.
      fileOnly: routes.filter((r) => r.mapped && !r.byPath).map((r) => r.path),
    },
    featureComponentDirs: {
      total: components.length,
      unmapped: components
        .filter((c) => !c.mapped && !c.excluded)
        .map((c) => c.dir),
      excluded: components
        .filter((c) => !c.mapped && c.excluded)
        .map((c) => c.dir),
    },
  };
}

function mapIdList() {
  const ids = [];
  for (const file of featureFiles()) {
    const text = readFileSync(join(mapDir, file), "utf8");
    const section =
      text.split(/^## /m).find((s) => s.startsWith("Sub-features")) ?? "";
    for (const m of section.matchAll(/^- `(F\d{2}\.[a-z0-9-]+)`:?\s*(.*)$/gm))
      ids.push({ id: m[1], file, summary: m[2].slice(0, 100) });
  }
  return ids;
}

// The map index (references/feature-map/README.md), home of the baseline line.
function indexText() {
  const path = join(mapDir, "README.md");
  return existsSync(path) ? readFileSync(path, "utf8") : "";
}

// The recorded baseline, with where it stands relative to HEAD.
// Two baseline lines (a merge kept both sides) are nobody's baseline.
function assertOneBaselineLine(index) {
  const lines = baselineLines(index);
  if (lines.length > 1)
    throw new CliError(
      `The map index has ${lines.length} Maintenance baseline lines.`,
      {
        code: 2,
        hint: "Remove the line that is not the latest pass's proposal (a merge kept both sides), then re-run.",
      },
    );
}

function baselineInfo() {
  const index = indexText();
  assertOneBaselineLine(index);
  const recorded = parseBaseline(index);
  if (!recorded) return null;
  let known = false;
  let ancestorOfHead = false;
  let commitsSince;
  try {
    execFileSync("git", ["cat-file", "-e", `${recorded.sha}^{commit}`], {
      cwd: repoRoot,
      stdio: "ignore",
    });
    known = true;
    ancestorOfHead =
      spawnSync("git", ["merge-base", "--is-ancestor", recorded.sha, "HEAD"], {
        cwd: repoRoot,
      }).status === 0;
    if (ancestorOfHead)
      commitsSince = Number(
        git(["rev-list", "--first-parent", "--count", `${recorded.sha}..HEAD`]),
      );
  } catch {
    // Not in this clone (shallow, or a SHA from another branch).
  }
  return { ...recorded, known, ancestorOfHead, commitsSince };
}

// Changed paths between two revisions (or from a file), mapped to families.
function mapAffected(flags) {
  let changed;
  let range;
  let baseSource = "--base";
  if (flags.paths && flags.paths !== true) {
    const text =
      String(flags.paths) === "-"
        ? readFileSync(0, "utf8")
        : readFileSync(resolve(String(flags.paths)), "utf8");
    changed = text
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);
  } else {
    // Without --base, the range starts at the maintenance baseline the map
    // index records (the previous pass's TARGET), so a daily pass needs no
    // memory of yesterday.
    let base =
      flags.base && flags.base !== true ? String(flags.base) : undefined;
    if (!base) {
      const index = indexText();
      assertOneBaselineLine(index);
      const recorded = parseBaseline(index);
      if (!recorded)
        usage(
          "map affected needs --base REF [--target REF] or --paths FILE|-: the map index has no `Maintenance baseline:` line to start from",
          "control-openhands map affected --base origin/main",
        );
      base = recorded.sha;
      baseSource = "map index baseline";
    }
    const target =
      flags.target && flags.target !== true ? String(flags.target) : "HEAD";
    try {
      changed = git(["diff", "--name-only", `${base}..${target}`])
        .split("\n")
        .filter(Boolean);
    } catch (error) {
      // Say which end is missing, and why: a shallow clone that lacks the
      // commit is deepened; a commit that is not an ancestor of the target
      // was recorded from a branch, or the target is not main.
      const known = (ref) =>
        spawnSync("git", ["cat-file", "-e", `${ref}^{commit}`], {
          cwd: repoRoot,
        }).status === 0;
      let reason;
      let hint;
      if (!known(base)) {
        reason = `${base} is not in this clone`;
        hint =
          "A shallow clone may not hold the baseline: fetch main and deepen until it is present (maintenance.md step 1).";
      } else if (!known(target)) {
        reason = `${target} is not in this clone`;
        hint = "Fetch the target ref first (git fetch origin main).";
      } else {
        reason = `${base} is not an ancestor of ${target}`;
        hint =
          baseSource === "map index baseline"
            ? "The recorded baseline is not on this history: it was set from a branch, or the target is not main. Pass --base with a commit on main, and fix the line with map baseline --set <main commit>."
            : "Pick a base that the target descends from.";
      }
      throw new CliError(`git diff ${base}..${target} failed: ${reason}.`, {
        code: 3,
        hint,
        extra: { git: String(error.message).split("\n")[0] },
      });
    }
    range = { base, target, baseSource };
  }
  const result = affectedFamilies(familyList(), changed);
  return {
    ...(range ? { range } : {}),
    changed: changed.length,
    families: result.families,
    // Shared code (API clients, hooks, stores, styles, i18n): widen to the
    // consumers rather than sampling one screen (maintenance.md step 3).
    shared: result.shared,
    // Under src/ but owned by no family's Source: line: a map gap, or a
    // Source: line to extend.
    unmapped: result.unmapped,
    nonUserFacing: result.nonUserFacing,
  };
}

async function cmdMap({ positional, flags }) {
  const [sub] = positional;
  if (sub === "check") {
    const only =
      flags.file && flags.file !== true
        ? basename(String(flags.file))
        : undefined;
    if (flags["fix-counts"] && !only) fixIndexCounts();
    const result = mapCheck({ only });
    if (only) {
      result.problems = result.problems.filter(
        (p) => p.startsWith(`${only}:`) || p.includes(`(also in ${only})`),
      );
    }
    out({ ok: result.problems.length === 0, ...result });
    if (result.problems.length) process.exitCode = 1;
    return;
  }
  if (sub === "coverage") {
    const result = mapCoverage();
    const ok =
      result.routes.unmapped.length === 0 &&
      result.featureComponentDirs.unmapped.length === 0;
    out({ ok, ...result });
    if (!ok) process.exitCode = 1;
    return;
  }
  if (sub === "ids") {
    const ids = mapIdList();
    out({ ok: true, count: ids.length, ids });
    return;
  }
  if (sub === "routes") {
    out({ ok: true, routes: routePaths() });
    return;
  }
  if (sub === "affected") {
    const result = mapAffected(flags);
    out({ ok: true, ...result });
    return;
  }
  if (sub === "baseline") {
    if (flags.set !== undefined) {
      // Move the line to a main commit's full SHA and committer date: what a
      // pass that finished its range does before opening its PR. The line
      // reads main@<sha>, so the commit must be on main: a pass's branch tip
      // (a merge commit, the PR's own fixes) never is.
      if (flags.set === true)
        usage(
          "map baseline --set needs the frozen TARGET (a commit on main)",
          'control-openhands map baseline --set "$TARGET"',
        );
      const ref = String(flags.set);
      let sha;
      let date;
      try {
        sha = git(["rev-parse", "--verify", "--quiet", `${ref}^{commit}`]);
        date = git(["show", "-s", "--format=%cs", sha]);
      } catch {
        throw new CliError(`${ref} is not a commit in this clone.`, {
          code: 2,
          hint: 'control-openhands map baseline --set "$TARGET" (a commit on main; fetch origin main first)',
        });
      }
      const mainRef = ["origin/main", "main"].find(
        (r) =>
          spawnSync("git", ["rev-parse", "--verify", "--quiet", r], {
            cwd: repoRoot,
          }).status === 0,
      );
      let warning;
      if (mainRef) {
        const onMain =
          spawnSync("git", ["merge-base", "--is-ancestor", sha, mainRef], {
            cwd: repoRoot,
          }).status === 0;
        if (!onMain && !flags.force)
          throw new CliError(`${sha.slice(0, 12)} is not on ${mainRef}.`, {
            code: 2,
            hint: 'The line records main@<sha>: pass the frozen TARGET (control-openhands map baseline --set "$TARGET"), not the pass branch\'s HEAD. --force records it anyway.',
          });
      } else {
        warning =
          "Neither origin/main nor main is in this clone, so the commit could not be checked against main.";
      }
      const path = join(mapDir, "README.md");
      const index = indexText();
      assertOneBaselineLine(index);
      const before = parseBaseline(index);
      let text;
      try {
        text = withBaseline(index, sha, date);
      } catch (error) {
        throw new CliError(error.message, {
          code: 2,
          hint: "Add the line after the index's intro paragraph: `Maintenance baseline: main@<sha> (<date>). The next maintenance pass starts from this commit; a pass proposes the next baseline in its PR, and merging that PR accepts it.`",
        });
      }
      writeFileSync(path, text);
      out({
        ok: true,
        baseline: { sha, date },
        previous: before,
        file: relative(repoRoot, path),
        ...(warning ? { warning } : {}),
        note: "Proposed in this checkout only: the baseline moves when the PR that carries this line merges.",
      });
      return;
    }
    const info = baselineInfo();
    out({
      ok: true,
      baseline: info,
      ...(info
        ? {}
        : {
            hint: "The map index has no `Maintenance baseline:` line; map affected then needs --base.",
          }),
    });
    return;
  }
  if (sub === "testids") {
    const result = resolveTestids(
      citedTestids(mapDir, featureFiles()),
      sourceLiterals(join(repoRoot, "src")),
    );
    const ok = !flags.strict || result.unresolved.length === 0;
    out({
      ok,
      ...result,
      note:
        result.unresolved.length === 0
          ? "every cited test id is accounted for in src/"
          : "unresolved ids are candidates for drift (a rename), or built in a way this check misses; confirm with a live drive before editing the map",
    });
    if (!ok) process.exitCode = 1;
    return;
  }
  usage(
    "Usage: control-openhands map check|coverage|ids|routes|affected|testids|baseline",
    "control-openhands map coverage",
  );
}

// ---------------------------------------------------------------------------
// Help and dispatch.
// ---------------------------------------------------------------------------
const HELP = {
  login: `control-openhands login

Public-mode runs only: types the run's session key into whichever prompt the UI
shows (the onboarding "Add a backend" step, or api-key-entry-screen, filling
Host Name when empty). Never prints the key. Local-mode runs need no login.
`,
  restart: `control-openhands restart [--rotate-key] [--timeout SEC]

Stops this run's launcher process group and starts it again with the same
state directory, port block and keys, then waits for readiness. It asks the
checkout's launcher again whether it serves VS Code and adds or drops the
editor port to match, with a warning; when a newly needed editor port is busy
it refuses before stopping anything. --rotate-key writes a new session key
first, so the browser's stored key becomes stale (public mode then shows the
API-key prompt again). Reload the browser afterwards.
`,
  service: `control-openhands service status
control-openhands service stop automation|agent-server|frontend

Stops one service inside this run's own process group so unavailable/backend-down
states can be driven through the UI. Bring everything back with restart.
`,
  status: `control-openhands status — ports, revision, launcher and browser liveness of the current run.
`,
  runs: `control-openhands runs — every run under $OH_VERIFY_HOME with liveness.
`,
  env: `control-openhands env — export lines (OH_VERIFY_RUN, OH_VERIFY_BASE_URL) for this run.
`,
  _: `control-openhands — drive a real, isolated Agent Canvas stack like a user.

Usage: control-openhands <command> [args] [--run <dir>]

Lifecycle
  launch        Build this checkout if needed, start an isolated stack + browser
  doctor        Read-only health check; run first and after any surprise
  status        Show the current run's ports, revision and liveness
  restart       Restart this run's stack in place (same state); --rotate-key for stale-key states
  service       status | stop automation|agent-server|frontend (drive backend-down states)
  stop          Stop this run's browser and launcher process group (keeps evidence)
  runs          List runs under $OH_VERIFY_HOME
  env           Print export lines for this run

Arrange (preconditions, never UI proof)
  api           Call this run's API with its session key (writes need --write)
  llm           show | set | preset deepseek | check — configure LLM profiles from an env key
  fixture       git-repo | git-remote | folder | image | file | tarball | skill | mcp-server — disposable fixtures in the run

Essential pathways (driven through the real UI)
  login         Public mode: enter the session key on the API-key screen
  onboard       Telemetry consent + onboarding modal (--skip, or walk it with --agent)
  workspace     open PATH — pick a folder through Open Workspace (folder browser)
  conversation  start --prompt "..." [--wait] [--workspace PATH] | send | wait | status | list | events | pause

Drive and observe
  browser       goto/click/fill/press/wait/text/snapshot/testids/screenshot/viewport/errors/...

Evidence and map
  evidence      add | list | report — the run's pass/fail/blocked/not-run ledger
  map           check | coverage | ids | routes | affected | testids | baseline — keep the feature map honest

Run state lives in $OH_VERIFY_HOME (default: $TMPDIR/openhands-verify); the
current run is the 'current' symlink there, or --run / $OH_VERIFY_RUN.
Output is JSON. Exit: 0 ok, 1 action failed, 2 usage, 3 environment/doctor.

Examples:
  control-openhands launch
  control-openhands doctor
  control-openhands llm preset deepseek          # reads DEEPSEEK_API_KEY
  control-openhands onboard --skip
  control-openhands conversation start --prompt "Create hello.py printing hi and run it" --wait
  control-openhands browser goto /settings/secrets
  control-openhands browser click 'testid=add-secret-button'
  control-openhands browser screenshot --feature F14.create --name form
  control-openhands stop
`,
  launch: `control-openhands launch [--public] [--new] [--print-run] [--port N | --port-from N] [--build auto|always|never] [--min-free-mb 2000]
                         [--no-browser] [--timeout SEC] [--sdk-path DIR | --sdk-ref REF | --sdk-version V]
                         [--automation-path DIR | --automation-ref REF | --automation-version V] [--run-id ID] [--vscode]

Builds the checkout when build/verify-revision.txt does not match HEAD, then
starts bin/agent-canvas.mjs with a private HOME, state dir, session key and
encryption key, on a free port block (ingress P, agent-server P+1, automation
P+2, static frontend P+3). The editor port (P+1001) is reserved, and listed in
ports, only when the checkout's launcher serves VS Code. Where that is opt-in
(#17660), --vscode (OH_CANVAS_ENABLE_VSCODE=true) turns it on; where the editor
is bundled (before #17660, or since #18048), it is always reserved and --vscode
changes nothing. Waits for authenticated settings, automation health and the
SPA, then starts the browser daemon.
Idempotent: an alive current run is reused unless --new is given.
--print-run prints only the run directory, for
  export OH_VERIFY_RUN=$(control-openhands launch --new --print-run)
Every other command uses --run, else $OH_VERIFY_RUN, else the only live run;
with several live runs it refuses to guess (several agents on one machine).
Backend versions and VS Code come only from the flags: OH_AGENT_SERVER_*,
OH_AUTOMATION_* and OH_CANVAS_ENABLE_VSCODE in your shell are not forwarded
(launch warns). An Agent Server below
config/defaults.json compatibility.minimumAgentServer never gets
past the UI's backend-compatibility gate.

Environment: OH_VERIFY_HOME, CONTROL_OPENHANDS_BROWSER (Chromium path),
CONTROL_OPENHANDS_BROWSER_ARGS (extra Chromium flags), CONTROL_OPENHANDS_HEADED=1,
CONTROL_OPENHANDS_TRUST_CA=<pem>[,<pem>] (pin a TLS-intercepting proxy's CAs),
CONTROL_OPENHANDS_IGNORE_HTTPS_ERRORS=1 (sandboxes with leaf-only interception; never in CI).

Examples:
  control-openhands launch
  control-openhands launch --public --new
  control-openhands launch --sdk-path ../software-agent-sdk --build always
`,
  doctor: `control-openhands doctor [--skip-ui]

Read-only. Checks launcher group and ports, served build revision, unauthenticated
rejection, authenticated settings, agent-server version vs config/defaults.json
(pin, and the UI's minimum Agent Server version), automation health, LLM configuration (info) and a throwaway-tab UI probe.
Exit 3 when any check fails.
`,
  stop: `control-openhands stop [--purge-private]

Stops the browser daemon and SIGTERMs the launcher's process group (SIGKILL
after 30 s), verifies every run port closed and counts evidence files.
--purge-private then deletes private/ (keys, state, logs) and recounts evidence.
`,
  api: `control-openhands api <METHOD> </path> [--data JSON|@file | --data-file F --content-type T] [--write] [--pick a.b.c] [--max-bytes N]

Sends the run's session key only to the run's own origin. Non-GET needs --write:
use it to arrange preconditions, never as proof that a UI path works.

Examples:
  control-openhands api GET /api/settings
  control-openhands api GET /api/conversations/search?limit=5
  control-openhands api DELETE /api/settings/secrets/QA_TMP --write
`,
  llm: `control-openhands llm show
control-openhands llm preset deepseek [--api-key-env DEEPSEEK_API_KEY | --api-key-file PATH]
control-openhands llm set --profile NAME --model MODEL (--api-key-env VAR | --api-key-file PATH) [--base-url URL] [--no-activate]
control-openhands llm check --model MODEL (--api-key-env VAR | --api-key-file PATH) [--base-url URL]

set/preset validate with a 1-token completion first (skip with --no-validate).
Keys are read from an environment variable or file, never from argv.
'preset deepseek' saves deepseek-flash (deepseek/deepseek-flash, activated) and
deepseek-pro (deepseek/deepseek-v4-pro). Prefer flash; it is cheaper.

Examples:
  DEEPSEEK_API_KEY=... control-openhands llm preset deepseek
  control-openhands llm set --profile alt --model deepseek/deepseek-v4-pro --api-key-env DEEPSEEK_API_KEY --no-activate
`,
  onboard: `control-openhands onboard [--skip] [--agent openhands|claude-code|codex|gemini-cli] [--analytics on|off]

Answers the telemetry consent form (analytics off by default) and then either
skips the onboarding modal (--skip) or walks it: choose agent → keep current LLM
settings → close at say-hello. Skipping is not proof that onboarding works.
`,
  conversation: `control-openhands conversation start --prompt TEXT [--wait] [--until STATES] [--timeout SEC] [--stay] [--workspace PATH]
control-openhands conversation wait ID [--until finished,idle] [--timeout SEC] [--fresh]
        (--fresh after sending a message: ignore the previous run's terminal status)
control-openhands conversation send ID --prompt TEXT [--wait]   follow-up message through the composer
control-openhands conversation pause ID   arrange an interrupted conversation (API, as local Stop Runtime; not UI proof)
control-openhands conversation status ID
control-openhands conversation list
control-openhands conversation events ID [--last N] [--kinds MessageEvent,ActionEvent] [--full]
        [--grep TEXT] [--from-start]
        (texts are cut at 160 chars with truncated:true; --full keeps up to 4000;
         --grep searches whole events, e.g. the SystemPromptEvent with --from-start;
         rows show activated skills and, for the system prompt, the tool names)

'start' types into the home composer (testid=chat-input), presses
testid=submit-button and returns the new /conversations/<id>. --stay uses the
current page's composer instead of navigating home first. --workspace PATH
first picks that folder through Open Workspace (see 'workspace open'), so the
conversation runs in it. 'wait' polls the
conversation's execution_status until one of --until (default: any terminal).

Examples:
  control-openhands conversation start --prompt "Create hello.py that prints hi and run it" --wait --timeout 300
  control-openhands conversation events <id> --last 10
`,
  fixture: `control-openhands fixture git-repo [--name qa-repo] [--remote https://github.com/qa-example/qa-repo.git]
        # git repo in <run>/workspace (README, src/calc.py, test); --remote only sets origin
control-openhands fixture git-remote [--name qa-remote]   # bare repo <run>/workspace/qa-remote.git to push to
control-openhands fixture mcp-server [--name qa-mcp]      # stdio MCP server (tool qa_echo); prints command and args
control-openhands fixture folder [--name qa-folder]
control-openhands fixture image [--name qa-image] [--width 160 --height 96]   # PNG under evidence/_fixtures
control-openhands fixture file [--name qa-note.txt] [--content TEXT]
control-openhands fixture tarball [--name qa-tarball]   # evidence/_fixtures/qa-tarball.tar.gz (main.py prints pong)
control-openhands fixture skill [--name qa-hello] [--repo qa-repo] [--trigger qa-ping] [--body TEXT]
        # SKILL.md in the run's private ~/.agents/skills, or in a fixture repo's .agents/skills
`,
  workspace: `control-openhands workspace open PATH|NAME [--stay]

NAME (no slash) resolves inside <run>/workspace, where 'fixture' creates repos.

Drives Home > Open Workspace > workspace dropdown > + Add Workspace, walks the
folder browser (folder-browser-up / folder-browser-entry-<name>) to PATH,
clicks Use and Launch. The composer then carries the workspace chip; follow
with 'conversation start --stay', or use 'conversation start --workspace PATH'.

Example:
  control-openhands fixture git-repo --name qa-repo
  control-openhands conversation start --workspace qa-repo --prompt "List the files" --wait
`,
  browser: `control-openhands browser <verb> [selector] [args]

Selectors: segments joined by ' >> ', each one of
  testid=ID   role=button[name="Save"][exact]   text=Partial   text="Exact"
  label=Name  placeholder=Text  title=Text  alt=Text  nth=0  has-text=Text
  visible     or any Playwright/CSS selector
Example: 'testid=add-secret-form >> testid=submit-button'

Verbs
  start | stop                          daemon lifecycle (launch starts it)
  goto <path> [--allow-external]        navigate (run origin only by default)
  reload | back | forward | url
  click|dblclick <sel> [--force] [--timeout MS] [--button left|middle|right]
        [--modifiers Control,Meta,Shift,Alt] [--position X,Y] (offset inside the element)
        [--expect-url REGEX] (wait for client-side navigation; click returns the old URL otherwise)
        [--expect-new-url REGEX] (wait for a URL that differs from the pre-click one)
        [--hover-first [MS]] (hover, settle, then click: hover-driven toggles)
        [--observe SEL [--observe-ms 3000]] (record transient states such as "Saving..."; SEL uses the same syntax)
  hover|focus|check|uncheck <sel>        (hidden <input> switches: click their label instead)
  mouse-click X Y [--button B]          click at viewport coordinates (backdrops, overlays)
  tooltip <sel>                          hover and return the role=tooltip text (or the title attribute)
  fill|type <sel> [<value> | --value-file F | --value-env VAR]   (prefer the file/env forms for keys)
  press <Key> [--selector S]            e.g. Escape, Enter, Control+k, Meta+k
  select <sel> <value> | upload <sel> <file...> | scroll [<sel>] [--by PX] [--x PX]
  upload-via <trigger-sel> <file...>     click a button/menu item and answer the real file chooser
  drop-files <sel> <file...> [--stage enter|over]   files dragged in from the desktop
  paste <sel> [--text T] [--file F]...   a paste event carrying text and/or files
  drag <sel> <target-sel> [--position before|after] | drag <sel> --by DX,DY [--steps N]
  choose <combobox-sel> <option label>   open an autocomplete, filter, click the option
  clipboard [--write TEXT]               read (or arrange) the clipboard; Copy buttons write it
  wait <sel> [--state visible|hidden|attached|detached] | wait-url <regex> | wait-text <text>
  text|value|count|visible|enabled|bbox <sel> | attr <sel> <name>
        (password values are masked in value, snapshot and testids; value --reveal shows one)
  snapshot [<sel>] [--max-lines N] [--feature ID --name N]   ARIA tree (saved as evidence)
  testids [<sel>] [--hidden] [--filter part]                 discover on-screen data-testid handles (--hidden adds hidden/off-screen)
  screenshot [<sel>] --feature ID --name N [--full-page]     PNG under evidence/<ID>/
  viewport desktop|phone|narrow|tablet|WxH                    1440x1000, 390x844, 320x700, 820x1180
  clock --offset-ms N | --system ISO|+MS | --fixed ISO|+MS    skew the page's clock (install before the goto
                                         whose page should see it; the server's clock is untouched)
  errors [--clear] [--all] [--app-only] [--no-warnings]        page/console/HTTP errors since last clear
                                         (--clear prints the list, then empties it: run it before the action)
  events [--kinds pageerror,dialog,download] [--last N]
  eval <js expression>                   read-only inspection; never mutate app state with it
  tabs | tab <i> | close-tab <i> | dialogs [--policy accept|dismiss]
  wait-tab <url-regex> [--timeout MS]   wait for a pop-up (window.open) whose URL matches; prints its index
  downloads [--last N] [--inspect] [--contains TEXT]   saved files; --inspect adds a text head or zip entry names
  storage [--session] [--values]         localStorage (or sessionStorage) keys; values only on request
                                         (keys, tokens and secrets inside values are redacted)
  media [--clear]                        media playback recorded since load (sound features)
  network [--external] [--filter REGEX] [--clear] [--last N]   requests with status and redacted query (rows under "recent")
                                         (privacy/telemetry checks; --clear as for errors)
  toasts [--history [--clear]]          toasts on screen now (with links); --history adds every
                                         status/alert text seen since this page loaded
  reset                                  fresh browser profile (first-run state); the stack keeps running
  scroll <sel> --by PX                   scroll the element's scrollable container (settings, panels)
  scroll <sel> | scroll --by PX          bring into view | wheel at the mouse position

Long waits: pass --timeout 5000 for elements that may never appear; defaults are 30 s.
Placeholders in map recipes use <angle-brackets> (for example <id> from conversation start).

Failures return {ok:false,error,hint,failureScreenshot} and exit 1.
`,
  evidence: `control-openhands evidence add --feature ID --result pass|fail|blocked|not-run
        [--entry "UI path"] [--expected TEXT] [--actual TEXT] [--artifact path[,path]] [--note TEXT]
control-openhands evidence retract --feature ID [--entry "UI path"] [--note why]   drop a wrong row from the report
control-openhands evidence list [--feature F05]
control-openhands evidence report [--baseline RUN_DIR|ledger.jsonl] [--json] > report.md

'report' renders the latest row per feature and entry point: a family count
table, then every check with fail and blocked rows first and the --note column
(where blocked rows name their missing prerequisite). --baseline compares with
another run's ledger and lists newly failing, newly passing and unchecked rows
first (a daily pass against yesterday's run). --json prints the same data.
`,
  map: `control-openhands map check [--file Fnn-name.md]   lint the map (or one entry, skipping index links): four H2s, unique IDs, links, known commands, index counts
control-openhands map check --fix-counts   rewrite the index's sub-feature counts and total from the files, then lint
control-openhands map coverage   routes in src/routes.ts and src/components/features/* dirs not yet mapped
control-openhands map ids        every sub-feature ID with its file
control-openhands map routes     the route registry as path → route module
control-openhands map affected [--base REF] [--target REF] | --paths FILE|-
                                 changed paths mapped to the families whose Source: lines own them,
                                 shared code to widen, src/ paths no family owns (map gaps) and
                                 non-user-facing paths; without --base
                                 the range starts at the index's Maintenance baseline line
control-openhands map baseline [--set TARGET [--force]]
                                 the recorded baseline (sha, date, whether HEAD descends from it and
                                 by how many commits); --set moves the line to TARGET's full SHA and
                                 committer date in this checkout, for the pass's PR. TARGET must be a
                                 commit on origin/main (the line reads main@<sha>); a pass branch's
                                 HEAD is refused unless --force
control-openhands map testids [--strict]
                                 test ids the map drives that no literal or prefix in src/ accounts for
                                 (a cheap drift check before launching; --strict exits 1 on any)

check also verifies every Source: path exists and the baseline line's shape.
`,
};

const COMMANDS = {
  launch: cmdLaunch,
  status: cmdStatus,
  doctor: cmdDoctor,
  stop: cmdStop,
  env: cmdEnv,
  runs: cmdRuns,
  api: cmdApi,
  llm: cmdLlm,
  login: cmdLogin,
  restart: cmdRestart,
  service: cmdService,
  onboard: cmdOnboard,
  conversation: cmdConversation,
  fixture: cmdFixture,
  workspace: cmdWorkspace,
  browser: cmdBrowser,
  evidence: cmdEvidence,
  map: cmdMap,
};

async function main() {
  // Global flags such as --run DIR may come before the command.
  let argv = process.argv.slice(2);
  const leading = [];
  while (argv[0]?.startsWith("--") && !["--help", "-h"].includes(argv[0])) {
    const takesValue =
      !argv[0].includes("=") &&
      argv[1] !== undefined &&
      !argv[1].startsWith("--");
    leading.push(...argv.slice(0, takesValue ? 2 : 1));
    argv = argv.slice(takesValue ? 2 : 1);
  }
  const [command, ...afterCommand] = argv;
  const rest = [...afterCommand, ...leading];
  if (
    !command ||
    command === "--help" ||
    command === "-h" ||
    command === "help"
  ) {
    process.stdout.write(HELP[rest[0]] ?? HELP._);
    return;
  }
  const parsed = parseArgs(rest);
  if (parsed.flags.help || parsed.flags.h) {
    process.stdout.write(HELP[command] ?? HELP._);
    return;
  }
  const handler = COMMANDS[command];
  if (!handler) usage(`Unknown command ${command}`, "control-openhands --help");
  await handler(parsed);
}

main().catch((error) => {
  if (error instanceof CliError) {
    out({
      ok: false,
      error: error.message,
      hint: error.hint,
      ...(error.extra ?? {}),
    });
    process.exitCode = error.code;
    return;
  }
  out({ ok: false, error: String(error?.stack ?? error) });
  process.exitCode = 1;
});
