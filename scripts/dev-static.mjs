/**
 * Static-frontend Development Stack
 *
 * Same as the default automation stack but serves a production build of the
 * frontend via `scripts/static-server.mjs` instead of the Vite dev server.
 * Designed for slow / flaky network situations (e.g. plane wifi)
 * where Vite's ~1000 individual module requests per page load are the
 * bottleneck. The static build collapses the frontend into ~50 hashed
 * chunks that all 304 cleanly on reload.
 *
 * Architecture (identical to dev-with-automation, only the frontend differs):
 *   ┌──────────────────────────────────────────────────────────────────────────┐
 *   │              http://localhost:8000 (Ingress Proxy)                       │
 *   │              /api/automation/* → Automation Backend                      │
 *   │              /api/*, /sockets  → Agent Server                            │
 *   │              /*                → Static Frontend                         │
 *   └──────────────────────────────────────────────────────────────────────────┘
 *          │                    │                         │
 *          ▼                    ▼                         ▼
 *   ┌─────────────┐    ┌───────────────┐         ┌──────────────────┐
 *   │ sirv-cli    │    │ Agent Server  │         │ Automation       │
 *   │ build/      │    │ (uvx) :18000  │         │ Backend (uvx)    │
 *   │ :3001       │    │               │         │ :18001           │
 *   └─────────────┘    └───────────────┘         └──────────────────┘
 *
 * Usage:
 *   npm run dev:static
 *   npm run dev:static -- --port 12000
 *   npm run dev:static -- --skip-build  # reuse an existing build/
 *   npm run dev:static -- --automation-ref feat/my-branch
 *
 * Environment variables (all optional, same as dev):
 *   - PORT: Ingress port (default: 8000)
 *   - OH_AUTOMATION_GIT_REF: Git ref for automation (default: main)
 *   - OH_AGENT_SERVER_GIT_REF: Git ref for agent-server
 *   - OH_SECRET_KEY: Session secret key
 */

import { spawn, spawnSync } from "node:child_process";
import { join, resolve, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { setTimeout as delay } from "node:timers/promises";
import process from "node:process";

import { buildFrontend } from "./static-build.mjs";
import {
  buildAgentServerCommand,
  buildSafeDevConfig,
  buildAgentServerEnv,
  formatMissingUvxGuidance,
  isPortBusy,
  releaseStaleConversationLeases,
} from "./dev-safe.mjs";
import {
  getProcessTreeSpawnOptions,
  isProcessRunning,
  resolveWindowsCommand,
  signalProcessTree,
} from "./dev-process-utils.mjs";
import {
  buildAgentServerAutomationEnv,
  buildAutomationCommand,
  buildAutomationTelemetryEnv,
  buildAutomationRuntimeServicesInfo,
  buildConfig,
} from "./dev-with-automation.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(__dirname, "..");

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
}

function logStep(step, message) {
  console.log(`${c.cyan}[${step}]${c.reset} ${message}`);
}

function logSuccess(message) {
  console.log(`${c.green}✓${c.reset} ${message}`);
}

function logError(message) {
  console.error(`${c.red}✗${c.reset} ${message}`);
}

// ═══════════════════════════════════════════════════════════════════════════
// CLI parsing
// ═══════════════════════════════════════════════════════════════════════════

export function parseArgs(argv = process.argv.slice(2)) {
  const config = {
    port: null,
    automationGitRef: null,
    automationRepo: null,
    skipBuild: false,
    verbose: false,
  };

  for (let i = 0; i < argv.length; i++) {
    switch (argv[i]) {
      case "-p":
      case "--port":
        config.port = parseInt(argv[++i], 10);
        break;
      case "--automation-ref":
        config.automationGitRef = argv[++i];
        break;
      case "--automation-repo":
        config.automationRepo = argv[++i];
        break;
      case "--skip-build":
        config.skipBuild = true;
        break;
      case "-v":
      case "--verbose":
        config.verbose = true;
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
Agent Canvas Static-frontend Development Stack

Runs the automation stack, but serves a production build of the
frontend via scripts/static-server.mjs. Use this when a remote or flaky network
makes Vite's per-module requests painful (e.g. ngrok or plane wifi).

USAGE:
  npm run dev:static [-- options]

OPTIONS:
  -p, --port <port>           Ingress port (default: 8000)
  --automation-ref <ref>      Git ref for automation backend (default: main)
  --automation-repo <url>     Git repo URL for automation
  --skip-build                Reuse existing build/ directory (faster restart)
  -v, --verbose               Show detailed output
  -h, --help                  Show this help

ENVIRONMENT VARIABLES:
  PORT                        Alternative to --port
  OH_AUTOMATION_GIT_REF       Alternative to --automation-ref
  OH_AGENT_SERVER_GIT_REF     Git ref for agent-server SDK
  OH_SECRET_KEY               Secret key for sessions

ACCESS POINTS:
  Main UI:      http://localhost:PORT/
  API Docs:     http://localhost:PORT/api/automation/docs

NOTES:
  • The build is produced once at startup. Edit the source and rerun this
    command (or rebuild with \`npm run build:app\`) to pick up changes.
  • The static server sends ETag headers, so reloads return 304s instead of
    refetching content — much friendlier on slow links.
`);
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

function checkPrerequisites() {
  logStep("1/3", "Checking prerequisites...");

  if (!commandExists("uvx")) {
    console.error(formatMissingUvxGuidance(projectRoot));
    process.exit(1);
  }
  logSuccess("uvx found");

  if (!commandExists("npm")) {
    logError("npm is required but not found");
    process.exit(1);
  }
  logSuccess("npm found");
}

// ═══════════════════════════════════════════════════════════════════════════
// Process Management
// ═══════════════════════════════════════════════════════════════════════════

const processes = new Map();
let shuttingDown = false;

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

  proc.stdout.on("data", (data) => {
    data
      .toString()
      .split("\n")
      .filter(Boolean)
      .forEach((line) => logService(name, line.trim(), color));
  });

  proc.stderr.on("data", (data) => {
    data
      .toString()
      .split("\n")
      .filter(Boolean)
      .forEach((line) => logService(name, line.trim(), c.yellow));
  });

  proc.on("error", (error) => {
    logError(`${name} failed to start: ${error.message}`);
  });

  proc.on("exit", (code) => {
    if (code !== 0 && code !== null && !shuttingDown) {
      logService(name, `Exited with code ${code}`, c.red);
    }
    processes.delete(name);
  });

  processes.set(name, proc);
  return proc;
}

async function waitForService(name, url, timeoutMs = 30000) {
  const start = Date.now();

  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url);
      if (res.ok) {
        logService(name, `Ready at ${url}`, c.green);
        return true;
      }
    } catch {
      // Keep trying
    }
    await delay(500);
  }

  logService(name, `Timeout waiting for ${url}`, c.red);
  return false;
}

// ═══════════════════════════════════════════════════════════════════════════
// Service Starters (agent-server + automation are byte-for-byte the same as
// dev-with-automation; the only difference is the frontend service.)
// ═══════════════════════════════════════════════════════════════════════════

function startAgentServer(config) {
  logService(
    "agent-server",
    `Starting on port ${config.agentServerPort}...`,
    c.blue,
  );

  const agentServerCmd = buildAgentServerCommand(process.env);
  logService("agent-server", `Using ${agentServerCmd.source}`, c.dim);

  const safeConfig = buildSafeDevConfig(config.canvasPath, {
    ...process.env,
    OH_CANVAS_SAFE_STATE_DIR: config.stateDir,
    OH_CANVAS_SAFE_BACKEND_PORT: config.agentServerPort.toString(),
    OH_CANVAS_SAFE_VSCODE_PORT: config.vscodePort.toString(),
  });

  const agentServerEnv = {
    ...buildAgentServerEnv(safeConfig),
    ...buildAgentServerAutomationEnv(config),
  };

  spawnService(
    "agent-server",
    agentServerCmd.command,
    [
      ...agentServerCmd.args,
      "--host",
      "0.0.0.0",
      "--port",
      String(config.agentServerPort),
    ],
    {
      cwd: safeConfig.workspacesPath,
      env: agentServerEnv,
      color: c.blue,
    },
  );
}

function buildAutomationBackendEnv(config, env = process.env) {
  // Both backends share the same session API key value.
  return {
    AUTOMATION_AGENT_SERVER_URL: `http://localhost:${config.agentServerPort}`,
    AUTOMATION_AGENT_SERVER_API_KEY: config.sessionApiKey,
    AUTOMATION_DB_URL: `sqlite+aiosqlite:///${join(config.stateDir, "automations.db")}`,
    AUTOMATION_BASE_URL: `http://localhost:${config.ingressPort}`,
    AUTOMATION_WORKSPACE_BASE: join(config.stateDir, "workspaces"),
    AUTOMATION_LOCAL_API_KEY: config.sessionApiKey,
    ...buildAutomationTelemetryEnv(env),
    AUTOMATION_CORS_ORIGINS: `http://localhost:${config.ingressPort},http://127.0.0.1:${config.ingressPort},http://localhost:3001,http://127.0.0.1:3001`,
    FILE_STORE: "local",
    LOCAL_STORAGE_PATH: join(config.stateDir, "storage"),
    OPENHANDS_SUPPRESS_BANNER: "1",
  };
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
      "0.0.0.0",
      "--port",
      config.autoBackendPort.toString(),
    ],
    {
      cwd: config.stateDir,
      env: buildAutomationBackendEnv(config),
      color: c.green,
    },
  );
}

function startStaticServer(config) {
  // Reuse `vitePort` as the upstream port name so the ingress route table
  // below stays identical to dev-with-automation.mjs.
  logService("static", `Starting on port ${config.vitePort}...`, c.magenta);

  // Mirror the proxy targets that vite.config.ts exposes in dev mode so that
  // hitting :3001 directly behaves like Vite's dev server (e.g. /server_info
  // is forwarded to the agent-server instead of falling back to the SPA
  // shell). Without this, /server_info on :3001 returns index.html.
  const staticServerScript = join(projectRoot, "scripts", "static-server.mjs");
  const runtimeServicesInfo = JSON.stringify(
    buildAutomationRuntimeServicesInfo({
      ...config,
      frontendKind: "static",
    }),
  );
  spawnService(
    "static",
    "node",
    [
      staticServerScript,
      "--dir",
      join(config.canvasPath, "build"),
      "--port",
      String(config.vitePort),
      ...(process.env.VITE_BASE_PATH
        ? ["--base-path", process.env.VITE_BASE_PATH]
        : []),
      // Inject the API key so the pre-built frontend can authenticate
      // to the agent-server without a baked-in VITE_SESSION_API_KEY.
      ...(config.sessionApiKey
        ? ["--session-api-key", config.sessionApiKey]
        : []),
      "--runtime-services-info",
      runtimeServicesInfo,
      "--route",
      `/api/automation=http://localhost:${config.autoBackendPort}`,
      "--route",
      `/api=http://localhost:${config.agentServerPort}`,
      "--route",
      `/sockets=http://localhost:${config.agentServerPort}`,
      "--route",
      `/server_info=http://localhost:${config.agentServerPort}`,
      "--route",
      `/health=http://localhost:${config.agentServerPort}`,
      "--route",
      `/ready=http://localhost:${config.agentServerPort}`,
      "--route",
      `/alive=http://localhost:${config.agentServerPort}`,
      "--route",
      `/docs=http://localhost:${config.agentServerPort}`,
      "--route",
      `/redoc=http://localhost:${config.agentServerPort}`,
      "--route",
      `/openapi.json=http://localhost:${config.agentServerPort}`,
    ],
    {
      cwd: config.canvasPath,
      color: c.magenta,
    },
  );
}

function startIngress(config) {
  logService("ingress", `Starting on port ${config.ingressPort}...`, c.yellow);

  const ingressScript = join(projectRoot, "scripts", "ingress.mjs");
  const runtimeServicesInfo = JSON.stringify(
    buildAutomationRuntimeServicesInfo({
      ...config,
      frontendKind: "static",
    }),
  );

  spawnService(
    "ingress",
    "node",
    [
      ingressScript,
      "--port",
      config.ingressPort.toString(),
      "--runtime-services-info",
      runtimeServicesInfo,
      "--route",
      `/api/automation=http://localhost:${config.autoBackendPort}`,
      "--route",
      `/api=http://localhost:${config.agentServerPort}`,
      "--route",
      `/sockets=http://localhost:${config.agentServerPort}`,
      "--route",
      `/server_info=http://localhost:${config.agentServerPort}`,
      "--route",
      `/health=http://localhost:${config.agentServerPort}`,
      "--route",
      `/ready=http://localhost:${config.agentServerPort}`,
      "--route",
      `/alive=http://localhost:${config.agentServerPort}`,
      "--route",
      `/docs=http://localhost:${config.agentServerPort}`,
      "--route",
      `/redoc=http://localhost:${config.agentServerPort}`,
      "--route",
      `/openapi.json=http://localhost:${config.agentServerPort}`,
      "--default",
      `http://localhost:${config.vitePort}`,
    ],
    {
      cwd: projectRoot,
      color: c.yellow,
    },
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Shutdown / Banner
// ═══════════════════════════════════════════════════════════════════════════

function shutdown() {
  if (shuttingDown) return;
  shuttingDown = true;

  console.log("");
  console.log(`${c.yellow}Shutting down...${c.reset}`);

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
    process.exit(0);
  }, 3000);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

function printBanner(config) {
  console.log("");
  console.log(
    `${c.green}${c.bold}╔══════════════════════════════════════════════════════════════╗${c.reset}`,
  );
  console.log(
    `${c.green}${c.bold}║${c.reset}  ${c.bold}Agent Canvas Static-frontend Stack${c.reset}                          ${c.green}${c.bold}║${c.reset}`,
  );
  console.log(
    `${c.green}${c.bold}╠══════════════════════════════════════════════════════════════╣${c.reset}`,
  );
  console.log(
    `${c.green}${c.bold}║${c.reset}                                                              ${c.green}${c.bold}║${c.reset}`,
  );
  console.log(
    `${c.green}${c.bold}║${c.reset}  Main UI:      ${c.cyan}http://localhost:${config.ingressPort}/${c.reset}`.padEnd(
      75,
    ) + `${c.green}${c.bold}║${c.reset}`,
  );
  console.log(
    `${c.green}${c.bold}║${c.reset}  API Docs:     ${c.cyan}http://localhost:${config.ingressPort}/api/automation/docs${c.reset}`.padEnd(
      75,
    ) + `${c.green}${c.bold}║${c.reset}`,
  );
  console.log(
    `${c.green}${c.bold}║${c.reset}                                                              ${c.green}${c.bold}║${c.reset}`,
  );
  console.log(
    `${c.green}${c.bold}╚══════════════════════════════════════════════════════════════╝${c.reset}`,
  );
  console.log("");
  console.log(`${c.dim}State directory: ${config.stateDir}${c.reset}`);
  console.log(
    `${c.dim}Frontend served from: ${join(config.canvasPath, "build")}${c.reset}`,
  );
  console.log(
    `${c.dim}Edit sources, then re-run \`npm run dev:static\` to rebuild.${c.reset}`,
  );
  console.log(`${c.dim}Press Ctrl+C to stop${c.reset}`);
  console.log("");
}

// ═══════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════

async function main() {
  const args = parseArgs();
  const config = await buildConfig(args);

  console.log("");
  console.log(
    `${c.cyan}${c.bold}Agent Canvas Static-frontend Development Stack${c.reset}`,
  );
  console.log("");

  // Setup phase (1/3)
  checkPrerequisites();

  // Ensure isolated state dirs (same as dev-with-automation).
  const { mkdirSync } = await import("node:fs");
  for (const dir of [
    config.stateDir,
    join(config.stateDir, "dev_conversations"),
    join(config.stateDir, "workspaces"),
    join(config.stateDir, "bash_events"),
    join(config.stateDir, "storage"),
  ]) {
    mkdirSync(dir, { recursive: true });
  }

  // Build phase (2/3): block until the SPA is ready to serve.
  buildFrontend(config, args);

  // Service phase (3/3)
  logStep("3/3", "Starting services...");

  // The agent-server skip-loads any conversation whose `owner_lease.json`
  // is held by a different `owner_instance_id` and not yet expired (45 s
  // TTL). If a previous agent-server (e.g. from `npm run dev`) was killed
  // ungracefully — or we restart faster than the lease TTL — every
  // conversation gets hidden until those stale leases age out, which
  // looks like "the new agent-server doesn't inherit my conversations".
  // Bail out if a live agent-server is already bound to our port (we'd
  // collide anyway), otherwise unlink the stale leases so the new server
  // can claim ownership immediately.
  if (await isPortBusy(config.agentServerPort)) {
    logError(
      `Port ${config.agentServerPort} is already in use — another ` +
        `agent-server is running. Stop it (e.g. quit \`npm run dev\`) ` +
        `before running dev:static.`,
    );
    process.exit(1);
  }
  const conversationsPath = join(config.stateDir, "dev_conversations");
  const cleared = releaseStaleConversationLeases(conversationsPath);
  if (cleared > 0) {
    logService(
      "agent-server",
      `Released ${cleared} stale conversation lease(s) so the new ` +
        `agent-server can resume ownership.`,
      c.dim,
    );
  }

  startAgentServer(config);
  await waitForService(
    "agent-server",
    `http://localhost:${config.agentServerPort}/server_info`,
  );

  startAutomationBackend(config);

  startStaticServer(config);

  await delay(2000);

  startIngress(config);

  await delay(1000);

  printBanner(config);
}

// ═══════════════════════════════════════════════════════════════════════════
// Exports for testing
// ═══════════════════════════════════════════════════════════════════════════

export { buildAutomationBackendEnv, buildFrontend, startStaticServer };

// ═══════════════════════════════════════════════════════════════════════════
// Main entry point (only when run directly, not when imported)
// ═══════════════════════════════════════════════════════════════════════════

const isMainModule =
  process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;

if (isMainModule) {
  main().catch((err) => {
    logError(`Fatal error: ${err.message}`);
    if (err.stack) {
      console.error(c.dim + err.stack + c.reset);
    }
    process.exit(1);
  });
}
