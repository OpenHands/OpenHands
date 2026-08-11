/**
 * Development Stack with Automation Service
 *
 * Extends agent-canvas's dev-safe.mjs to additionally run the OpenHands Automation
 * backend via uvx. No cloning required - runs directly from git reference.
 *
 * Uses a standalone ingress proxy to route traffic to multiple backends.
 *
 * Architecture:
 *   ┌──────────────────────────────────────────────────────────────────────────┐
 *   │              http://localhost:8000 (Ingress Proxy)                       │
 *   │              /api/automation/* → Automation Backend                      │
 *   │              /api/*, /sockets  → Agent Server                            │
 *   │              /*                → Vite Dev Server                         │
 *   └──────────────────────────────────────────────────────────────────────────┘
 *          │                    │                         │
 *          ▼                    ▼                         ▼
 *   ┌─────────────┐    ┌───────────────┐         ┌──────────────────┐
 *   │ Vite        │    │ Agent Server  │         │ Automation       │
 *   │ :3001       │    │ (uvx) :18000  │         │ Backend (uvx)    │
 *   │             │    │               │         │ :18001           │
 *   └─────────────┘    └───────────────┘         └──────────────────┘
 *
 * Usage:
 *   node scripts/dev-with-automation.mjs
 *   node scripts/dev-with-automation.mjs --automation-ref feat/my-branch
 *   node scripts/dev-with-automation.mjs --port 12000
 *
 * Environment variables:
 *   - PORT: Ingress port (default: 8000)
 *   - OH_AUTOMATION_GIT_REF: Git ref for automation (default: main)
 *   - OH_AGENT_SERVER_LOCAL_PATH: Absolute path to a local software-agent-sdk
 *     checkout. Highest precedence for agent-server source selection: rebuilds
 *     the agent-server from local source and installs openhands-sdk,
 *     openhands-tools and openhands-workspace as editable so source edits are
 *     picked up without manual reinstall.
 *   - OH_AGENT_SERVER_GIT_REF: Git ref for agent-server
 * Secrets:
 *   The session API key is automatically seeded into agent-server secrets
 *   as OPENHANDS_AUTOMATION_API_KEY, making it available to agents in conversations.
 *   Both the agent-server and automation backend use the same key value
 *   and the same `X-Session-API-Key` header for authentication.
 *   AUTOMATION_KV_SECRET is derived from the session key if not set explicitly,
 *   enabling the KV store out of the box for local development.
 */

import { spawn, spawnSync } from "node:child_process";
import { mkdirSync, existsSync, readFileSync } from "node:fs";
import { join, resolve, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { homedir } from "node:os";
import { setTimeout as delay } from "node:timers/promises";
import process from "node:process";

import {
  assertPortsFree,
  buildAgentServerCommand,
  buildSafeDevConfig,
  buildAgentServerEnv,
  buildNpmScriptCommand,
  buildRuntimeServicesInfo,
  formatMissingUvxGuidance,
  validateFrontendDependencies,
  validateLocalAgentServerPath,
} from "./dev-safe.mjs";
import {
  createShutdownHookRegistry,
  getProcessTreeSpawnOptions,
  isProcessRunning,
  resolveWindowsCommand,
  signalProcessTree,
} from "./dev-process-utils.mjs";
import { fileLog, stripAnsi } from "./logger.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(__dirname, "..");

// ── Centralized config (single source of truth for versions, ports, etc.) ───
const SHARED_DEFAULTS = JSON.parse(
  readFileSync(join(projectRoot, "config", "defaults.json"), "utf-8"),
);

const DEFAULT_AUTOMATION_REPO = "https://github.com/OpenHands/automation";
const DEFAULT_AUTOMATION_PACKAGE = SHARED_DEFAULTS.packages.automation;
const DEFAULT_AUTOMATION_VERSION = SHARED_DEFAULTS.versions.automation;
const DEFAULT_AUTOMATION_SDK_VERSION = SHARED_DEFAULTS.versions.agentServer;
const DEFAULT_BACKEND_PORT = SHARED_DEFAULTS.ports.agentServer;
const DEFAULT_AUTOMATION_PORT = SHARED_DEFAULTS.ports.automation;
const DEFAULT_POSTHOG_API_KEY = SHARED_DEFAULTS.telemetry.posthogApiKey;
const DEFAULT_POSTHOG_HOST = SHARED_DEFAULTS.telemetry.posthogHost;

// ═══════════════════════════════════════════════════════════════════════════
// Terminal Styling
// ═══════════════════════════════════════════════════════════════════════════

const c = {
  reset: "\x1b[0m",
  bold: "\x1b[1m",
  dim: "\x1b[2m",
  red: "\x1b[31m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  blue: "\x1b[34m",
  magenta: "\x1b[35m",
  cyan: "\x1b[36m",
};

function logService(name, message, color = c.reset) {
  const ts = new Date().toISOString().split("T")[1].split(".")[0];
  console.log(`${c.dim}${ts}${c.reset} ${color}[${name}]${c.reset} ${message}`);
  fileLog("info", `[${name}] ${stripAnsi(message)}`);
}

function logStep(step, message) {
  console.log(`${c.cyan}[${step}]${c.reset} ${message}`);
  fileLog("info", `[${step}] ${message}`);
}

function logSuccess(message) {
  console.log(`${c.green}✓${c.reset} ${message}`);
  fileLog("info", `✓ ${message}`);
}

function logError(message) {
  console.error(`${c.red}✗${c.reset} ${message}`);
  fileLog("error", `✗ ${stripAnsi(message)}`);
}

/**
 * Parse one JSON log line produced by the SDK's JsonFormatter and return a
 * single-line human-readable string + an appropriate ANSI color.
 *
 * Returns null for non-JSON lines so callers can fall back to the raw text.
 *
 * @param {string} rawLine
 * @returns {{ text: string; color: string } | null}
 */
function parseAgentServerLogLine(rawLine) {
  try {
    const obj = JSON.parse(rawLine);
    if (!obj.levelname || obj.message === undefined) return null;
    const level = obj.levelname.padEnd(8);
    const location =
      obj.filename && obj.lineno ? `  ${obj.filename}:${obj.lineno}` : "";
    const text = `${level} ${obj.message}${location}`;
    const lvl = obj.levelname;
    const color =
      lvl === "DEBUG"
        ? c.dim
        : lvl === "WARNING"
          ? c.yellow
          : lvl === "ERROR" || lvl === "CRITICAL"
            ? c.red
            : c.blue;
    return { text, color };
  } catch {
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════

function parseArgs() {
  const args = process.argv.slice(2);
  const config = {
    port: null,
    automationGitRef: null,
    automationRepo: null,
    verbose: false,
    static: false,
    dynamic: false,
    staticDir: null,
    skipBuild: false,
    public: false,
    frontendOnly: false,
    backendOnly: false,
  };

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "-p":
      case "--port":
        config.port = parseInt(args[++i], 10);
        break;
      case "--automation-ref":
        config.automationGitRef = args[++i];
        break;
      case "--automation-repo":
        config.automationRepo = args[++i];
        break;
      case "-v":
      case "--verbose":
        config.verbose = true;
        break;
      case "--static":
        config.static = true;
        break;
      case "--dynamic":
        config.dynamic = true;
        break;
      case "--static-dir":
        config.staticDir = args[++i];
        break;
      case "--skip-build":
        config.skipBuild = true;
        break;
      case "--public":
        config.public = true;
        break;
      case "--frontend-only":
        config.frontendOnly = true;
        break;
      case "--backend-only":
        config.backendOnly = true;
        break;
      case "-h":
      case "--help":
        showHelp();
        process.exit(0);
    }
  }

  return config;
}

function showHelp() {
  console.log(`
Agent Canvas + Automation Development Stack

Runs agent-canvas with the automation backend (via uvx, no clone needed).
Uses a standalone ingress proxy to route traffic.

USAGE:
  node scripts/dev-with-automation.mjs [options]

OPTIONS:
  -p, --port <port>           Ingress port (default: 8000)
  --automation-ref <ref>      Git ref for automation (branch/tag/SHA)
  --automation-repo <url>     Git repo URL (default: ${DEFAULT_AUTOMATION_REPO})
  --static                    Serve an existing production build instead of Vite
  --static-dir <dir>          Static build directory (default: build/)
  --skip-build                Reuse build/ when the launcher builds static assets
  --dynamic                   Force Vite dev server when a wrapper defaults static
  --frontend-only             Start only the frontend behind ingress
  --backend-only              Start only agent-server + automation behind ingress
  -v, --verbose               Show detailed output
  -h, --help                  Show this help

ENVIRONMENT VARIABLES:
  PORT                        Alternative to --port
  OH_AUTOMATION_GIT_REF       Git ref for automation (overrides default version)
  OH_AUTOMATION_VERSION       Specific PyPI version for automation (default: ${DEFAULT_AUTOMATION_VERSION})
  OH_AGENT_SERVER_LOCAL_PATH  Absolute path to a local software-agent-sdk checkout (highest precedence)
  OH_AGENT_SERVER_GIT_REF     Git ref for agent-server SDK (overrides default version)
  OH_AGENT_SERVER_VERSION     Specific PyPI version for agent-server
  OH_SECRET_KEY               Secret key for sessions

SECRETS:
  The session API key is automatically seeded into agent-server secrets
  as OPENHANDS_AUTOMATION_API_KEY, making it available to agents in conversations.
  Both backends (agent-server and automation) share the same key value.
  AUTOMATION_KV_SECRET defaults to the session key so the KV store works
  out of the box; override with an explicit value for stronger isolation.

ACCESS POINTS:
  Main UI:      http://localhost:PORT/
  API Docs:     http://localhost:PORT/api/automation/docs
`);
}

/**
 * Build the uvx command for running automation backend.
 *
 * Environment variables (highest precedence first):
 * - OH_AUTOMATION_GIT_REF: Git commit SHA or branch name
 * - OH_AUTOMATION_VERSION: Specific PyPI version (e.g., "1.0.0a1")
 *
 * If none are set, defaults to the released version specified by
 * DEFAULT_AUTOMATION_VERSION. Set OH_AUTOMATION_GIT_REF to use a
 * git branch or commit instead.
 */
function buildAutomationCommand(env = process.env) {
  const gitRef = env.OH_AUTOMATION_GIT_REF;
  const version = env.OH_AUTOMATION_VERSION;
  const repoUrl = env.OH_AUTOMATION_REPO || DEFAULT_AUTOMATION_REPO;

  const uvxArgs = [];
  let source = "";

  if (gitRef) {
    // Use git ref - refresh to ensure latest commit is fetched
    const gitUrl = `git+${repoUrl}@${gitRef}`;
    uvxArgs.push(
      "--refresh",
      "--from",
      gitUrl,
      "uvicorn",
      "openhands.automation.app:app",
    );
    source = `git (${gitRef})`;
  } else if (version) {
    // Use specific PyPI version
    uvxArgs.push(
      "--from",
      `${DEFAULT_AUTOMATION_PACKAGE}==${version}`,
      "uvicorn",
      "openhands.automation.app:app",
    );
    source = `PyPI (${version})`;
  } else {
    // Default to released PyPI version
    uvxArgs.push(
      "--from",
      `${DEFAULT_AUTOMATION_PACKAGE}==${DEFAULT_AUTOMATION_VERSION}`,
      "uvicorn",
      "openhands.automation.app:app",
    );
    source = `PyPI (${DEFAULT_AUTOMATION_VERSION}, default)`;
  }

  return {
    command: "uvx",
    args: uvxArgs,
    source,
  };
}

async function buildConfig(args, env = process.env) {
  // Apply args to env for buildAutomationCommand
  if (args.automationGitRef) {
    env.OH_AUTOMATION_GIT_REF = args.automationGitRef;
  }
  if (args.automationRepo) {
    env.OH_AUTOMATION_REPO = args.automationRepo;
  }

  const frontendOnly = Boolean(args.frontendOnly);
  const backendOnly = Boolean(args.backendOnly);
  if (frontendOnly && backendOnly) {
    throw new Error(
      "--frontend-only and --backend-only cannot be used together",
    );
  }

  const launchFrontend = !backendOnly;
  const launchAgentServer = !frontendOnly;
  const launchAutomation = !frontendOnly;
  const isPublic = args.public;

  if (isPublic && frontendOnly) {
    throw new Error("--public cannot be used with --frontend-only");
  }

  // In public mode, LOCAL_BACKEND_API_KEY is required — without it the
  // auth screen has nothing to validate against.
  if (isPublic && !env.LOCAL_BACKEND_API_KEY) {
    logError(
      "PUBLIC MODE requires LOCAL_BACKEND_API_KEY environment variable.\n" +
        "  Example: LOCAL_BACKEND_API_KEY=my-secret npm run dev -- --public",
    );
    process.exit(1);
  }

  // Preferred ports (from env or defaults).
  // OH_CANVAS_SAFE_BACKEND_PORT / OH_CANVAS_SAFE_AUTOMATION_PORT /
  // OH_CANVAS_SAFE_VITE_PORT allow tests (and advanced users) to redirect
  // internal service ports without affecting the production default.
  const preferredIngressPort = args.port || parseInt(env.PORT, 10) || 8000;
  const preferredBackendPort =
    parseInt(env.OH_CANVAS_SAFE_BACKEND_PORT, 10) || DEFAULT_BACKEND_PORT;
  const preferredAutomationPort =
    parseInt(env.OH_CANVAS_SAFE_AUTOMATION_PORT, 10) || DEFAULT_AUTOMATION_PORT;
  const preferredVitePort = parseInt(env.OH_CANVAS_SAFE_VITE_PORT, 10) || 3001;

  // Fail fast if any preferred port for a service in this mode is already in use.
  const requiredPorts = [{ name: "ingress", port: preferredIngressPort }];
  if (launchAgentServer) {
    requiredPorts.push({ name: "agent-server", port: preferredBackendPort });
  }
  if (launchAutomation) {
    requiredPorts.push({ name: "automation", port: preferredAutomationPort });
  }
  if (launchFrontend) {
    requiredPorts.push({ name: "frontend", port: preferredVitePort });
  }

  logStep("ports", "Checking ports...");
  await assertPortsFree(requiredPorts);

  const vscodePort = preferredBackendPort + 1000;

  // API key — shared by both agent-server and automation backend.
  // Both validate it via the `X-Session-API-Key` header.
  // LOCAL_BACKEND_API_KEY is the single user-facing env var: if set it's
  // used directly; otherwise one is auto-generated and persisted.
  const stateDir =
    env.OH_CANVAS_SAFE_STATE_DIR ||
    join(homedir(), ".openhands", "agent-canvas");

  const safeConfig = buildSafeDevConfig(projectRoot, {
    ...env,
    OH_CANVAS_SAFE_STATE_DIR: stateDir,
    OH_CANVAS_SAFE_BACKEND_PORT: preferredBackendPort.toString(),
    OH_CANVAS_SAFE_VSCODE_PORT: vscodePort.toString(),
  });
  const sessionApiKey = safeConfig.sessionApiKey;

  if (isPublic) {
    logService(
      "auth",
      "PUBLIC MODE — key will NOT be injected into the frontend",
      c.yellow,
    );
    logService(
      "auth",
      "Users must paste the LOCAL_BACKEND_API_KEY in the browser",
      c.dim,
    );
  }

  return {
    // Ingress port (main entry point)
    ingressPort: preferredIngressPort,

    // Service ports (internal)
    agentServerPort: preferredBackendPort,
    autoBackendPort: preferredAutomationPort,
    vitePort: preferredVitePort,
    vscodePort,

    // Paths
    canvasPath: projectRoot,

    // Data directories (same as dev-safe.mjs)
    stateDir,
    // Only bake the host-side workspace path when this launcher also starts
    // the agent-server that can read it. In frontend-only mode the backend may
    // be a tunnel/remote service, so leave VITE_WORKING_DIR unset unless the
    // user explicitly supplied a backend-relative value.
    viteWorkingDir: launchAgentServer
      ? safeConfig.workingDir
      : env.VITE_WORKING_DIR,

    // Auth — single key for both backends
    sessionApiKey,

    // Public mode — the session key should NOT be baked into the frontend
    isPublic,

    frontendOnly,
    backendOnly,
    launchFrontend,
    launchAgentServer,
    launchAutomation,

    verbose: args.verbose,
  };
}

// ═══════════════════════════════════════════════════════════════════════════
// Prerequisites & Setup
// ═══════════════════════════════════════════════════════════════════════════

function commandExists(cmd) {
  const result =
    process.platform === "win32"
      ? spawnSync("where.exe", [cmd], { stdio: "pipe" })
      : spawnSync("sh", ["-c", `command -v ${cmd}`], { stdio: "pipe" });

  return result.status === 0;
}

function checkPrerequisites({
  checkUvx = true,
  checkNpm = true,
  checkFrontendDependencies = true,
} = {}) {
  logStep("1/2", "Checking prerequisites...");

  if (checkUvx) {
    if (!commandExists("uvx")) {
      const uvxGuidance = formatMissingUvxGuidance(projectRoot);
      console.error(uvxGuidance);
      fileLog("error", stripAnsi(uvxGuidance));
      process.exit(1);
    }
    logSuccess("uvx found");
  }

  if (checkNpm) {
    if (!commandExists("npm")) {
      logError("npm is required but not found");
      process.exit(1);
    }
    logSuccess("npm found");
  }

  if (checkFrontendDependencies) {
    try {
      validateFrontendDependencies(projectRoot);
    } catch (error) {
      logError(error instanceof Error ? error.message : String(error));
      process.exit(1);
    }
    logSuccess("frontend dependencies found");
  }
}

function ensureDirectories(config) {
  const dirs = [
    config.stateDir,
    // Both agent-server and automation use storage; create it unconditionally
    // whenever either backend service runs (i.e. not frontend-only).
    ...(!config.frontendOnly ? [join(config.stateDir, "storage")] : []),
  ];

  if (config.launchAgentServer) {
    dirs.push(
      join(config.stateDir, "dev_conversations"),
      join(config.stateDir, "workspaces"),
      join(config.stateDir, "bash_events"),
    );
  }

  if (config.launchAutomation) {
    dirs.push(
      // Automation DB directory — matches docker/entrypoint.sh mkdir -p behaviour.
      dirname(
        join(dirname(config.stateDir), SHARED_DEFAULTS.paths.automationDb),
      ),
    );
  }

  for (const dir of dirs) {
    mkdirSync(dir, { recursive: true });
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Process Management
// ═══════════════════════════════════════════════════════════════════════════

const processes = new Map();
const shutdownHooks = createShutdownHookRegistry((err) => {
  logService("cleanup", `Cleanup hook failed: ${err.message}`, c.yellow);
});

// Optional external listener for every service log line. Set by `main()` from
// its `onServiceLog` option so embedded launchers (e.g. the Electron desktop
// app) can stream uvx download / install progress to their loading window
// without touching the terminal logging path. Receives `(name, line, level)`
// where `level` is one of "stdout" | "stderr" | "info" | "warn" | "error".
let serviceLogListener = null;

export function setServiceLogListener(listener) {
  serviceLogListener = typeof listener === "function" ? listener : null;
}

function emitServiceLog(name, line, level) {
  if (!serviceLogListener) return;
  try {
    serviceLogListener(name, line, level);
  } catch {
    // Never let a listener bug crash the dev stack.
  }
}

function registerShutdownHook(hook) {
  return shutdownHooks.add(hook);
}

function spawnService(name, command, args, options = {}) {
  const proc = spawn(
    resolveWindowsCommand(command),
    args,
    getProcessTreeSpawnOptions({
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env, ...options.env },
      cwd: options.cwd,
    }),
  );

  const color = options.color || c.reset;
  const parseLogLine = options.parseLogLine;

  proc.stdout.on("data", (data) => {
    data
      .toString()
      .split("\n")
      .filter(Boolean)
      .forEach((line) => {
        const trimmed = line.trim();
        const parsed = parseLogLine ? parseLogLine(trimmed) : null;
        logService(
          name,
          parsed ? parsed.text : trimmed,
          parsed ? parsed.color : color,
        );
        emitServiceLog(name, trimmed, "stdout");
      });
  });

  proc.stderr.on("data", (data) => {
    data
      .toString()
      .split("\n")
      .filter(Boolean)
      .forEach((line) => {
        const trimmed = line.trim();
        const parsed = parseLogLine ? parseLogLine(trimmed) : null;
        logService(
          name,
          parsed ? parsed.text : trimmed,
          parsed ? parsed.color : c.yellow,
        );
        emitServiceLog(name, trimmed, "stderr");
      });
  });

  proc.on("error", (error) => {
    logError(`${name} failed to start: ${error.message}`);
    emitServiceLog(name, `failed to start: ${error.message}`, "error");
  });

  proc.on("exit", (code, _signal) => {
    if (code !== 0 && code !== null && !shuttingDown) {
      logService(name, `Exited with code ${code}`, c.red);
      emitServiceLog(name, `exited with code ${code}`, "error");
    }
    processes.delete(name);
  });

  processes.set(name, proc);
  return proc;
}

async function waitForService(name, url, timeoutMs = 30000) {
  const start = Date.now();
  let lastError = null;

  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url, { signal: AbortSignal.timeout(5000) });
      if (res.ok) {
        logService(name, `Ready at ${url}`, c.green);
        return true;
      }
    } catch (err) {
      lastError = err;
      // Keep trying
    }
    await delay(500);
  }

  const elapsed = Math.round((Date.now() - start) / 1000);
  logService(name, `Timeout waiting for ${url} after ${elapsed}s`, c.red);
  if (lastError) {
    logService(name, `Last error: ${lastError.message}`, c.dim);
  }
  return false;
}

// ═══════════════════════════════════════════════════════════════════════════
// Service Starters
// ═══════════════════════════════════════════════════════════════════════════

const AUTOMATION_ROUTE_PREFIX = "/api/automation";
const AGENT_SERVER_ROUTE_PREFIXES = [
  "/api",
  "/sockets",
  "/server_info",
  "/health",
  "/ready",
  "/alive",
  "/docs",
  "/redoc",
  "/openapi.json",
];

function getLocalServiceRoutes(config) {
  const routes = [];

  // These services bind to IPv4 loopback, but localhost can resolve to ::1.
  if (config.launchAutomation) {
    routes.push([
      AUTOMATION_ROUTE_PREFIX,
      `http://127.0.0.1:${config.autoBackendPort}`,
    ]);
  }

  if (config.launchAgentServer) {
    for (const prefix of AGENT_SERVER_ROUTE_PREFIXES) {
      routes.push([prefix, `http://127.0.0.1:${config.agentServerPort}`]);
    }
  }

  return routes;
}

function buildRouteArgs(routes) {
  return routes.flatMap(([prefix, url]) => ["--route", `${prefix}=${url}`]);
}

/**
 * Build --reject-prefix args for the static server.
 * In frontend-only mode, API paths that have no backend should return 503
 * instead of being SPA-fallbacked to index.html.
 */
function getRejectPrefixes(config) {
  const prefixes = [];
  if (!config.launchAutomation) {
    prefixes.push(AUTOMATION_ROUTE_PREFIX);
  }
  if (!config.launchAgentServer) {
    for (const prefix of AGENT_SERVER_ROUTE_PREFIXES) {
      prefixes.push(prefix);
    }
  }
  return prefixes;
}

function buildRejectPrefixArgs(prefixes) {
  return prefixes.flatMap((prefix) => ["--reject-prefix", prefix]);
}

function getFrontendBackend(config) {
  return config.launchFrontend ? `http://localhost:${config.vitePort}` : null;
}

function buildViteBackendEnv(config, env = process.env) {
  const backendBaseUrl = config.launchAgentServer
    ? `http://127.0.0.1:${config.ingressPort}`
    : (env.VITE_BACKEND_BASE_URL ?? "http://127.0.0.1:8000");
  const backendHost = config.launchAgentServer
    ? `127.0.0.1:${config.ingressPort}`
    : (env.VITE_BACKEND_HOST ?? new URL(backendBaseUrl).host);

  return {
    VITE_BACKEND_HOST: backendHost,
    VITE_BACKEND_BASE_URL: backendBaseUrl,
  };
}

function buildAgentServerAutomationEnv(config) {
  return {
    // Make the session API key available to terminal commands spawned by the
    // agent-server as OPENHANDS_AUTOMATION_API_KEY. The launcher also seeds
    // this into Settings > Secrets, but agents commonly create automations
    // with a curl command that references `$OPENHANDS_AUTOMATION_API_KEY`;
    // exposing it here keeps that path working even before/without
    // secret-registry env expansion.
    OPENHANDS_AUTOMATION_API_KEY: config.sessionApiKey,
  };
}

function buildAutomationTelemetryEnv(env = process.env) {
  const telemetryDisabled = env.VITE_DO_NOT_TRACK === "1";
  const apiKey =
    env.AUTOMATION_POSTHOG_API_KEY ||
    env.VITE_POSTHOG_API_KEY ||
    (telemetryDisabled ? "" : DEFAULT_POSTHOG_API_KEY);

  if (!apiKey) return {};

  return {
    AUTOMATION_POSTHOG_API_KEY: apiKey,
    AUTOMATION_POSTHOG_HOST:
      env.AUTOMATION_POSTHOG_HOST ||
      env.VITE_POSTHOG_HOST ||
      DEFAULT_POSTHOG_HOST,
  };
}

function startAgentServer(config) {
  logService(
    "agent-server",
    `Starting on port ${config.agentServerPort}...`,
    c.blue,
  );

  const agentServerCmd = buildAgentServerCommand(process.env);
  logService("agent-server", `Using ${agentServerCmd.source}`, c.dim);

  // Build safe config for agent-server env vars
  const safeConfig = buildSafeDevConfig(config.canvasPath, {
    ...process.env,
    OH_CANVAS_SAFE_STATE_DIR: config.stateDir,
    OH_CANVAS_SAFE_BACKEND_PORT: config.agentServerPort.toString(),
    OH_CANVAS_SAFE_VSCODE_PORT: config.vscodePort.toString(),
  });

  const agentServerEnv = {
    ...buildAgentServerEnv(safeConfig),
    ...buildAgentServerAutomationEnv(config),
    OPENHANDS_REMOTE_WS_READY_REQUIRED:
      process.env.OPENHANDS_REMOTE_WS_READY_REQUIRED || "false",
    // Ensure the agent-server uses the resolved key from config. This is
    // LOCAL_BACKEND_API_KEY when set, or the auto-generated persisted key.
    OH_SESSION_API_KEYS_0: config.sessionApiKey,
    // Emit structured JSON log lines instead of Rich-formatted output.
    // Rich wraps long messages across multiple lines and prepends its own
    // timestamp; LOG_JSON=true produces one JSON object per record which
    // parseAgentServerLogLine re-formats into a clean single-line entry.
    LOG_JSON: "true",
  };

  spawnService(
    "agent-server",
    agentServerCmd.command,
    [
      ...agentServerCmd.args,
      "--host",
      "127.0.0.1",
      "--port",
      String(config.agentServerPort),
    ],
    {
      cwd: safeConfig.workspacesPath,
      env: agentServerEnv,
      color: c.blue,
      parseLogLine: parseAgentServerLogLine,
    },
  );
}

function startAutomationBackend(config) {
  logService(
    "automation",
    `Starting on port ${config.autoBackendPort}...`,
    c.green,
  );

  const automationCmd = buildAutomationCommand(process.env);
  logService("automation", `Using ${automationCmd.source}`, c.dim);

  spawnService(
    "automation",
    automationCmd.command,
    [
      ...automationCmd.args,
      "--host",
      "127.0.0.1",
      "--port",
      config.autoBackendPort.toString(),
    ],
    {
      cwd: config.stateDir,
      env: {
        // Force UTF-8 for all Python file I/O (same reason as agent-server;
        // see buildAgentServerEnv in dev-safe.mjs).
        PYTHONUTF8: "1",
        OPENHANDS_REMOTE_WS_READY_REQUIRED:
          process.env.OPENHANDS_REMOTE_WS_READY_REQUIRED || "false",
        // The URL the automation backend itself uses to call the
        // agent-server's REST API (tarball upload + bash dispatch).
        //
        // Priority:
        //   1. AUTOMATION_AGENT_SERVER_URL explicitly set in the user's env
        //   2. `localhost:<agentServerPort>`
        AUTOMATION_AGENT_SERVER_URL:
          process.env.AUTOMATION_AGENT_SERVER_URL ||
          `http://localhost:${config.agentServerPort}`,
        // The URL exported into the in-sandbox bash chain as
        // `AGENT_SERVER_URL` (read by main.py / setup.sh to call back into
        // the agent-server).
        //
        // Priority:
        //   1. AUTOMATION_SANDBOX_AGENT_SERVER_URL explicitly set in env
        //   2. launcher-provided value
        //   3. unset — backend falls back to AUTOMATION_AGENT_SERVER_URL
        ...(process.env.AUTOMATION_SANDBOX_AGENT_SERVER_URL ||
        config.sandboxAgentServerUrl
          ? {
              AUTOMATION_SANDBOX_AGENT_SERVER_URL:
                process.env.AUTOMATION_SANDBOX_AGENT_SERVER_URL ||
                config.sandboxAgentServerUrl,
            }
          : {}),
        AUTOMATION_AGENT_SERVER_API_KEY: config.sessionApiKey,
        // ~/.openhands/automation/automations.db — matches docker/entrypoint.sh.
        AUTOMATION_DB_URL: `sqlite+aiosqlite:///${join(dirname(config.stateDir), SHARED_DEFAULTS.paths.automationDb)}`,
        // The automation backend uses this as its publicly-reachable base
        // URL: it's appended to callback URLs and injected into each
        // sandbox as `AUTOMATION_API_URL` (consumed by setup.sh for
        // /sdk-version and by the SDK for run completion).
        // Priority:
        //   1. AUTOMATION_BASE_URL explicitly set in the user's env
        //   2. launcher-provided host
        //   3. `localhost`
        AUTOMATION_BASE_URL:
          process.env.AUTOMATION_BASE_URL ||
          `http://${config.automationApiHost ?? "localhost"}:${config.ingressPort}`,
        // The dispatcher resolves this path and embeds it into a
        // `mkdir -p ...` shell command executed by the agent-server.
        // Priority:
        //   1. AUTOMATION_WORKSPACE_BASE explicitly set in the user's env
        //   2. `automationWorkspaceBase` option passed by the launcher
        //   3. host-side default under config.stateDir
        AUTOMATION_WORKSPACE_BASE:
          process.env.AUTOMATION_WORKSPACE_BASE ||
          config.automationWorkspaceBase ||
          join(config.stateDir, "workspaces"),
        // Session API key for self-hosted auth — shared with agent-server via X-Session-API-Key header
        AUTOMATION_LOCAL_API_KEY: config.sessionApiKey,
        ...buildAutomationTelemetryEnv(),
        // KV store secret — required for automations to use the built-in
        // key-value store for state persistence between runs. Used for JWT
        // signing and value encryption.
        // Priority:
        //   1. AUTOMATION_KV_SECRET explicitly set in the user's env
        //   2. sessionApiKey — convenient zero-config default for local dev
        AUTOMATION_KV_SECRET:
          process.env.AUTOMATION_KV_SECRET || config.sessionApiKey,
        // CORS: allow localhost origins for dev, unless explicitly overridden.
        AUTOMATION_CORS_ORIGINS:
          process.env.AUTOMATION_CORS_ORIGINS ||
          `http://localhost:${config.ingressPort},http://127.0.0.1:${config.ingressPort},http://localhost:3001,http://127.0.0.1:3001`,
        FILE_STORE: "local",
        LOCAL_STORAGE_PATH: join(config.stateDir, "storage"),
        OPENHANDS_SUPPRESS_BANNER: "1",
      },
      color: c.green,
    },
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

let shuttingDown = false;

function shutdown() {
  if (shuttingDown) return;
  shuttingDown = true;

  console.log("");
  console.log(`${c.yellow}Shutting down...${c.reset}`);
  fileLog("info", "Shutting down...");

  for (const [name, proc] of processes) {
    logService(name, "Stopping...", c.dim);
    signalProcessTree(proc, "SIGTERM");
  }

  setTimeout(() => {
    for (const [name, proc] of processes) {
      if (isProcessRunning(proc)) {
        logService(name, "Force stopping...", c.dim);
        signalProcessTree(proc, "SIGKILL");
      }
    }
    shutdownHooks.run();
    process.exit(0);
  }, 3000);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
process.on("SIGHUP", shutdown);

function startIngress(config) {
  logService("ingress", `Starting on port ${config.ingressPort}...`, c.yellow);

  const ingressScript = join(projectRoot, "scripts", "ingress.mjs");
  const frontendBackend = getFrontendBackend(config);
  const runtimeServicesInfo = config.launchAgentServer
    ? JSON.stringify(buildAutomationRuntimeServicesInfo(config))
    : null;

  spawnService(
    "ingress",
    "node",
    [
      ingressScript,
      "--port",
      config.ingressPort.toString(),
      ...(runtimeServicesInfo
        ? ["--runtime-services-info", runtimeServicesInfo]
        : []),
      ...buildRouteArgs(getLocalServiceRoutes(config)),
      ...(frontendBackend ? ["--default", frontendBackend] : []),
    ],
    {
      cwd: projectRoot,
      color: c.yellow,
    },
  );
}

/**
 * Build the JSON-serializable runtime services info for an automation
 * stack. Backend-serving processes append this to `/server_info` so any
 * frontend connected to the backend can populate the agent's
 * `<RUNTIME_SERVICES>` system-prompt block.
 */
export function buildAutomationRuntimeServicesInfo(config) {
  return buildRuntimeServicesInfo({
    mode: config.mode ?? "dev:automation",
    agentHostAlias: config.agentHostAlias ?? "localhost",
    agentServerPort: config.agentServerPort,
    ingressPort: config.ingressPort,
    frontendPort: config.launchFrontend ? config.vitePort : undefined,
    // The same port hosts Vite in dynamic mode and a static-file server
    // in static mode. The launcher records this on the config so the
    // description shown to the agent matches reality.
    frontendKind: config.frontendKind ?? "vite",
    automation: config.launchAutomation
      ? { port: config.autoBackendPort }
      : undefined,
  });
}

function startVite(config) {
  logService("vite", `Starting on port ${config.vitePort}...`, c.magenta);

  const frontendCommand = buildNpmScriptCommand("dev:frontend");

  const viteEnv = {
    // Full-stack mode points Vite at this launcher's ingress. Frontend-only
    // mode uses the separately running backend ingress instead.
    ...buildViteBackendEnv(config),
    VITE_FRONTEND_PORT: config.vitePort.toString(),
  };
  if (config.viteWorkingDir) {
    viteEnv.VITE_WORKING_DIR = config.viteWorkingDir;
  }

  // In local mode, bake the session key into the frontend so the user
  // never has to paste it. In public mode, omit the key and set
  // VITE_AUTH_REQUIRED so the frontend shows the API key entry screen
  // immediately (no network round-trip needed).
  if (config.launchAgentServer && config.isPublic) {
    viteEnv.VITE_AUTH_REQUIRED = "true";
  } else if (config.launchAgentServer) {
    viteEnv.VITE_SESSION_API_KEY = config.sessionApiKey;
  }

  spawnService("vite", frontendCommand.command, frontendCommand.args, {
    cwd: config.canvasPath,
    env: viteEnv,
    color: c.magenta,
  });
}

/**
 * Seed the session API key into agent-server's secrets store as
 * OPENHANDS_AUTOMATION_API_KEY so agents can authenticate with the
 * automation backend in curl commands during conversations.
 *
 * Includes retry logic to handle slow server startup or transient failures.
 *
 * @param {object} config - Configuration object with agentServerPort, sessionApiKey
 * @param {object} options - Options for retry behavior
 * @param {number} options.maxRetries - Maximum number of retry attempts (default: 5)
 * @param {number} options.retryDelayMs - Delay between retries in ms (default: 2000)
 * @param {number} options.timeoutMs - Request timeout in ms (default: 10000)
 * @returns {Promise<boolean>} True if seeding succeeded, false otherwise
 */
async function seedAutomationSecret(config, options = {}) {
  const { maxRetries = 5, retryDelayMs = 2000, timeoutMs = 10000 } = options;

  const secretName = "OPENHANDS_AUTOMATION_API_KEY";
  const secretDescription =
    "API key for authenticating with the automation backend";

  logService("secrets", `Seeding ${secretName} into agent-server...`, c.dim);

  const url = `http://localhost:${config.agentServerPort}/api/settings/secrets`;
  const body = JSON.stringify({
    name: secretName,
    value: config.sessionApiKey,
    description: secretDescription,
  });

  const headers = {
    "Content-Type": "application/json",
    // Include session API key if configured
    ...(config.sessionApiKey && { "X-Session-API-Key": config.sessionApiKey }),
  };

  let lastError = null;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch(url, {
        method: "PUT",
        headers,
        body,
        signal: AbortSignal.timeout(timeoutMs),
      });

      if (response.ok) {
        logService("secrets", `${secretName} seeded successfully`, c.green);
        return true;
      }

      const text = await response.text();
      lastError = `HTTP ${response.status}: ${text}`;

      // Don't retry on authentication errors - they won't resolve with retries
      if (response.status === 401 || response.status === 403) {
        logService(
          "secrets",
          `Warning: Failed to seed secret (${response.status}): ${text}`,
          c.yellow,
        );
        return false;
      }

      // Retry on server errors or service unavailable
      if (attempt < maxRetries) {
        logService(
          "secrets",
          `Retry ${attempt}/${maxRetries} after ${response.status}...`,
          c.dim,
        );
        await delay(retryDelayMs);
      }
    } catch (err) {
      lastError = err.message;

      // Connection errors likely mean server isn't ready - wait and retry
      if (attempt < maxRetries) {
        logService(
          "secrets",
          `Retry ${attempt}/${maxRetries}: ${err.message}`,
          c.dim,
        );
        await delay(retryDelayMs);
      }
    }
  }

  logService(
    "secrets",
    `Warning: Failed to seed secret after ${maxRetries} attempts: ${lastError}`,
    c.yellow,
  );
  return false;
}

function printBanner(config) {
  const stackName = config.frontendOnly
    ? "Agent Canvas Frontend Stack"
    : config.backendOnly
      ? "Agent Canvas Backend Stack"
      : "Agent Canvas + Automation Stack";

  // padEnd counts invisible ANSI escape bytes as visible characters, so we
  // compute the visible length separately and pad with spaces accordingly.
  const ansiEscape = String.fromCharCode(27);
  const ansiRe = new RegExp(`${ansiEscape}\\[[0-9;]*m`, "g");
  const ansiPadEnd = (str, targetVisible) => {
    const visible = str.replace(ansiRe, "").length;
    return str + " ".repeat(Math.max(0, targetVisible - visible));
  };
  // The box has 62-char inner width; each content line needs 63 visible chars
  // before the trailing border (1 leading ║ + 62 inner).
  const BOX_INNER = 63;

  console.log("");
  console.log(
    `${c.green}${c.bold}╔══════════════════════════════════════════════════════════════╗${c.reset}`,
  );
  console.log(
    ansiPadEnd(
      `${c.green}${c.bold}║${c.reset}  ${c.bold}${stackName}${c.reset}`,
      BOX_INNER,
    ) + `${c.green}${c.bold}║${c.reset}`,
  );
  console.log(
    `${c.green}${c.bold}╠══════════════════════════════════════════════════════════════╣${c.reset}`,
  );
  console.log(
    `${c.green}${c.bold}║${c.reset}                                                              ${c.green}${c.bold}║${c.reset}`,
  );
  console.log(
    ansiPadEnd(
      `${c.green}${c.bold}║${c.reset}  Ingress:      ${c.cyan}http://localhost:${config.ingressPort}/${c.reset}`,
      BOX_INNER,
    ) + `${c.green}${c.bold}║${c.reset}`,
  );
  if (config.launchFrontend) {
    console.log(
      ansiPadEnd(
        `${c.green}${c.bold}║${c.reset}  Main UI:      ${c.cyan}http://localhost:${config.ingressPort}/${c.reset}`,
        BOX_INNER,
      ) + `${c.green}${c.bold}║${c.reset}`,
    );
  }
  if (config.launchAutomation) {
    console.log(
      ansiPadEnd(
        `${c.green}${c.bold}║${c.reset}  API Docs:     ${c.cyan}http://localhost:${config.ingressPort}/api/automation/docs${c.reset}`,
        BOX_INNER,
      ) + `${c.green}${c.bold}║${c.reset}`,
    );
  }
  console.log(
    `${c.green}${c.bold}║${c.reset}                                                              ${c.green}${c.bold}║${c.reset}`,
  );
  console.log(
    `${c.green}${c.bold}╚══════════════════════════════════════════════════════════════╝${c.reset}`,
  );
  console.log("");
  console.log(`${c.dim}State directory: ${config.stateDir}${c.reset}`);
  console.log(`${c.dim}Press Ctrl+C to stop${c.reset}`);
  console.log("");

  // Write a compact plain-text summary to the log file.
  const summary = [
    `${stackName} — started`,
    `  Ingress:         http://localhost:${config.ingressPort}/`,
    ...(config.launchFrontend
      ? [`  Main UI:         http://localhost:${config.ingressPort}/`]
      : []),
    ...(config.launchAutomation
      ? [
          `  API Docs:        http://localhost:${config.ingressPort}/api/automation/docs`,
        ]
      : []),
    `  State directory: ${config.stateDir}`,
  ];
  fileLog("info", summary.join("\n"));
}

async function main(options = {}) {
  const {
    bannerTitle = "Agent Canvas + Automation Development Stack",
    startAgentServer: startAgentServerOverride,
    extraPrereqs,
    viteWorkingDir,
    // Path used as `AUTOMATION_WORKSPACE_BASE` by the automation backend.
    // Defaults to a host-side path under config.stateDir.
    automationWorkspaceBase,
    // Host used in `AUTOMATION_BASE_URL` (the URL the automation sandbox
    // uses to call back into the automation backend). Defaults to `localhost`.
    automationApiHost,
    // Value exported as `AUTOMATION_SANDBOX_AGENT_SERVER_URL` to the
    // automation backend. This is the URL the in-sandbox bash chain uses
    // to reach the agent-server. When unset the backend falls back to
    // AUTOMATION_AGENT_SERVER_URL.
    sandboxAgentServerUrl,
    staticMode: staticModeOverride,
    defaultStaticMode = false,
    buildStaticFrontend,
    staticDir: staticDirOverride,
    // Hostname the agent uses to reach services running on the host.
    agentHostAlias = "localhost",
    // Human-readable label for the dev mode, surfaced in the agent's
    // <RUNTIME_SERVICES> system-prompt block.
    mode = "dev:automation",
    // When true, enable public mode (require LOCAL_BACKEND_API_KEY,
    // don't bake session key into frontend).
    isPublic: isPublicOverride,
    // When true, skip the npm prerequisite check. Used by the Electron desktop
    // launcher where npm is not needed at runtime in static mode.
    skipNpmCheck = false,
    // How long to wait for the agent-server's `/server_info` to return 200
    // before continuing. Defaults to 60 s, which is fine for warm-cache dev
    // workflows. The Electron desktop launcher bumps this to several minutes
    // because first-launch on a fresh machine runs `uvx` to download Python
    // and install `openhands-agent-server` from PyPI, which can take much
    // longer than 60 s on a slow network.
    agentServerReadyTimeoutMs = 60_000,
    // Optional `(name, line, level)` callback that receives every service log
    // line (stdout, stderr, and lifecycle events) emitted by any spawned
    // backend process. Used by the Electron loading screen to surface uvx
    // download / install progress to the user. `level` is one of
    // "stdout" | "stderr" | "info" | "warn" | "error".
    onServiceLog,
  } = options;

  // Install the listener early so log lines emitted before the first
  // `spawnService` call (e.g. by future setup steps) are also captured.
  setServiceLogListener(onServiceLog);

  const args = parseArgs();

  // Allow options to override CLI args for public mode
  if (isPublicOverride != null) {
    args.public = isPublicOverride;
  }

  // Allow options to override CLI args (for bin/agent-canvas.mjs)
  const useStaticMode =
    staticModeOverride ??
    (args.dynamic ? false : args.static || defaultStaticMode);
  const staticDir =
    staticDirOverride ?? args.staticDir ?? join(projectRoot, "build");

  const modeLabel = useStaticMode && !args.backendOnly ? "(Static)" : "";
  const titleWithMode = modeLabel ? `${bannerTitle} ${modeLabel}` : bannerTitle;

  console.log("");
  console.log(`${c.cyan}${c.bold}${titleWithMode}${c.reset}`);
  console.log("");
  fileLog("info", titleWithMode);

  // Setup phase
  checkPrerequisites({
    checkUvx: !args.frontendOnly,
    // Static-mode + backend-only has no frontend to build, so npm is not
    // required — unless the caller provides a custom buildStaticFrontend hook.
    // The Electron desktop launcher passes `skipNpmCheck: true` because the
    // packaged binary serves a pre-built static frontend and never invokes
    // npm at runtime, so we suppress the check unconditionally there.
    checkNpm:
      !skipNpmCheck &&
      ((!useStaticMode && !args.backendOnly) ||
        typeof buildStaticFrontend === "function"),
    checkFrontendDependencies:
      (!useStaticMode && !args.backendOnly) ||
      typeof buildStaticFrontend === "function",
  });

  // Fail fast on an obviously bad OH_AGENT_SERVER_LOCAL_PATH so we don't waste
  // time allocating ports / generating keys / launching uvx with a path that
  // would only produce a cryptic build error. Mirrors dev-safe.mjs and
  // dev-extra-backend.mjs.
  if (!args.frontendOnly && process.env.OH_AGENT_SERVER_LOCAL_PATH) {
    try {
      validateLocalAgentServerPath(process.env.OH_AGENT_SERVER_LOCAL_PATH);
    } catch (error) {
      logError(error instanceof Error ? error.message : String(error));
      process.exit(1);
    }
  }

  // Build config with dynamic port allocation
  const config = await buildConfig(args);
  if (viteWorkingDir) config.viteWorkingDir = viteWorkingDir;
  if (automationWorkspaceBase) {
    config.automationWorkspaceBase = automationWorkspaceBase;
  }
  if (automationApiHost) {
    config.automationApiHost = automationApiHost;
  }
  if (sandboxAgentServerUrl) {
    config.sandboxAgentServerUrl = sandboxAgentServerUrl;
  }
  // Stamp the dev-mode label, host alias, and frontend kind on the config
  // so downstream helpers (Vite spawn, static build) can produce a
  // runtime-services info object describing what the agent can reach.
  config.mode = mode;
  config.agentHostAlias = agentHostAlias;
  config.frontendKind = useStaticMode ? "static" : "vite";
  ensureDirectories(config);
  if (typeof extraPrereqs === "function") {
    extraPrereqs(config);
  }

  if (
    config.launchFrontend &&
    useStaticMode &&
    typeof buildStaticFrontend === "function"
  ) {
    buildStaticFrontend(config, args);
  }

  // In static mode, verify build exists after any launcher-managed build.
  if (config.launchFrontend && useStaticMode && !existsSync(staticDir)) {
    logError(`Static directory not found: ${staticDir}`);
    logError(`Run 'npm run build' first to create the static files.`);
    process.exit(1);
  }

  // Start services phase
  logStep("2/2", "Starting services...");

  let agentServerReady = false;

  // 1. Start agent-server first (automation depends on it).
  //
  // Readiness timeout defaults to 60 s, which is fine for `npm run dev` against
  // a warm uvx cache. The Electron desktop launcher overrides this via the
  // `agentServerReadyTimeoutMs` option because first-launch on a fresh machine
  // runs `uvx` to download Python + install `openhands-agent-server` from PyPI,
  // which can take several minutes. Dropping the user into a half-booted UI
  // before that completes triggers axios "Request timeout" popups on the first
  // SPA fetch that hits an unbound port 18000.
  if (config.launchAgentServer) {
    const agentServerStarter = startAgentServerOverride ?? startAgentServer;
    agentServerStarter(config);

    agentServerReady = await waitForService(
      "agent-server",
      `http://localhost:${config.agentServerPort}/server_info`,
      agentServerReadyTimeoutMs,
    );
  }

  // 2. Seed automation API key into agent-server secrets
  // This makes the key available to agents during conversations
  // Note: seedAutomationSecret has its own retry logic if server is still warming up
  if (config.launchAutomation && agentServerReady) {
    await seedAutomationSecret(config);
  } else if (config.launchAutomation) {
    logService(
      "secrets",
      "Skipping secret seeding - agent-server not ready",
      c.yellow,
    );
  }

  // 3. Start automation backend
  if (config.launchAutomation) {
    startAutomationBackend(config);
  }

  // 4. Start frontend server (Vite dev server OR static server)
  if (config.launchFrontend) {
    if (useStaticMode) {
      startStaticFrontend(config, staticDir);
    } else {
      startVite(config);
    }
  }

  // 5. Wait for services to be ready
  await delay(2000);

  // 6. Start ingress proxy (routes traffic only to running services)
  startIngress(config);

  // Wait for ingress to start
  await delay(1000);

  printBanner(config);

  // Return the resolved config + readiness signal so embedded launchers can
  // (a) build URLs from the actual allocated ports and (b) decide whether to
  // show an error to the user when the agent-server never came up.
  return { config, agentServerReady };
}

function startStaticFrontend(config, staticDir) {
  logService("static", `Starting on port ${config.vitePort}...`, c.magenta);
  logService("static", `Serving from: ${staticDir}`, c.dim);

  // Build the runtime-services info JSON so static-server can append it to
  // /server_info. The static-server also injects the old window global for
  // compatibility with previously built frontend bundles.
  const runtimeServicesInfo = config.launchAgentServer
    ? JSON.stringify(buildAutomationRuntimeServicesInfo(config))
    : null;

  const staticServerScript = join(projectRoot, "scripts", "static-server.mjs");
  spawnService(
    "static",
    "node",
    [
      staticServerScript,
      "--dir",
      staticDir,
      "--port",
      String(config.vitePort),
      ...(process.env.VITE_BASE_PATH
        ? ["--base-path", process.env.VITE_BASE_PATH]
        : []),
      // In local mode, inject the API key so the pre-built frontend can
      // authenticate transparently. In public mode, pass --auth-required
      // so the frontend shows the API key entry screen instead.
      ...(config.launchAgentServer && !config.isPublic && config.sessionApiKey
        ? ["--session-api-key", config.sessionApiKey]
        : []),
      ...(config.launchAgentServer && config.isPublic
        ? ["--auth-required"]
        : []),
      // Inject runtime-services info so the agent knows what's reachable.
      ...(runtimeServicesInfo
        ? ["--runtime-services-info", runtimeServicesInfo]
        : []),
      // Proxy routes only to services that this launch mode started.
      ...buildRouteArgs(getLocalServiceRoutes(config)),
      // Reject known API prefixes that have no backend — returns 503
      // instead of SPA-fallbacking to index.html.
      ...buildRejectPrefixArgs(getRejectPrefixes(config)),
    ],
    {
      cwd: config.canvasPath,
      color: c.magenta,
    },
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Exports for testing
// ═══════════════════════════════════════════════════════════════════════════

export {
  buildAgentServerAutomationEnv,
  buildAutomationCommand,
  buildAutomationTelemetryEnv,
  buildConfig,
  buildRouteArgs,
  buildViteBackendEnv,
  getFrontendBackend,
  getLocalServiceRoutes,
  main,
  registerShutdownHook,
  spawnService,
  commandExists,
  logService,
  logStep,
  logSuccess,
  logError,
  c,
  DEFAULT_AUTOMATION_REPO,
  DEFAULT_AUTOMATION_PACKAGE,
  DEFAULT_AUTOMATION_VERSION,
  DEFAULT_AUTOMATION_SDK_VERSION,
  DEFAULT_BACKEND_PORT,
  DEFAULT_AUTOMATION_PORT,
};

// ═══════════════════════════════════════════════════════════════════════════
// Main entry point (only when run directly, not when imported)
// ═══════════════════════════════════════════════════════════════════════════

// Check if this module is the main entry point
const isMainModule =
  process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;

if (isMainModule) {
  main().catch((err) => {
    logError(`Fatal error: ${err.message}`);
    if (err.stack) {
      console.error(c.dim + err.stack + c.reset);
      fileLog("error", err.stack);
    }
    process.exit(1);
  });
}
