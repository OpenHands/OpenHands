import { spawn } from "node:child_process";
import { mkdirSync } from "node:fs";
import path from "node:path";
import process from "node:process";
import { setTimeout as delay } from "node:timers/promises";
import { pathToFileURL } from "node:url";

import {
  buildAgentServerCommand,
  buildAgentServerEnv,
  buildSafeDevConfig,
  formatMissingUvxGuidance,
  validateLocalAgentServerPath,
} from "./dev-safe.mjs";
import {
  getProcessTreeSpawnOptions,
  isProcessRunning,
  signalProcessTree,
} from "./dev-process-utils.mjs";

const DEFAULT_EXTRA_BACKEND_PORT = 18002;
const DEFAULT_EXTRA_VSCODE_PORT = 18003;
const DEFAULT_WAIT_TIMEOUT_MS = 30_000;

function parsePort(value, fallback) {
  if (value == null || value === "") {
    return fallback;
  }

  const parsed = Number.parseInt(value, 10);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`Invalid port: ${value}`);
  }

  return parsed;
}

/**
 * Build a config for an *extra* standalone agent-server that shares the
 * bundled instance's persistence (state dir, conversations, secret key)
 * but listens on a different backend + vscode port.
 *
 * @param {string} cwd
 * @param {Record<string, string | undefined>} env
 */
export function buildExtraBackendConfig(cwd = process.cwd(), env = process.env) {
  const base = buildSafeDevConfig(cwd, env);

  const backendPort = parsePort(
    env.OH_CANVAS_EXTRA_BACKEND_PORT,
    DEFAULT_EXTRA_BACKEND_PORT,
  );
  const vscodePort = parsePort(
    env.OH_CANVAS_EXTRA_VSCODE_PORT,
    DEFAULT_EXTRA_VSCODE_PORT,
  );

  return {
    ...base,
    backendPort,
    vscodePort,
    backendBaseUrl: `http://127.0.0.1:${backendPort}`,
    backendHost: `127.0.0.1:${backendPort}`,
  };
}

function isEnoentError(error) {
  return Boolean(
    (error &&
      typeof error === "object" &&
      "code" in error &&
      error.code === "ENOENT") ||
      /ENOENT/.test(String(error)),
  );
}

async function waitForServer(url, timeoutMs = DEFAULT_WAIT_TIMEOUT_MS) {
  const startedAt = Date.now();

  while (Date.now() - startedAt < timeoutMs) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        return;
      }
    } catch {
      // Keep polling until timeout.
    }

    await delay(500);
  }

  throw new Error(`Timed out waiting for agent-server at ${url}`);
}

function spawnProcess(command, args, options = {}) {
  const child = spawn(command, args, getProcessTreeSpawnOptions({
    stdio: "inherit",
    ...options,
  }));

  child.once("error", (error) => {
    if (isEnoentError(error) && command === "uvx") {
      console.error(formatMissingUvxGuidance(options?.cwd));
    } else if (isEnoentError(error)) {
      console.error(
        `Failed to start ${command}. Make sure it is installed and on your PATH.`,
      );
    } else {
      console.error(`Failed to start ${command}:`, error);
    }
  });

  return child;
}

async function main() {
  const config = buildExtraBackendConfig();

  if (process.env.OH_AGENT_SERVER_LOCAL_PATH) {
    validateLocalAgentServerPath(process.env.OH_AGENT_SERVER_LOCAL_PATH);
  }

  for (const dir of [
    config.stateDir,
    config.tmuxTmpDir,
    config.conversationsPath,
    config.workspacesPath,
    config.bashEventsDir,
  ]) {
    mkdirSync(dir, { recursive: true });
  }

  const agentServerCmd = buildAgentServerCommand();

  const secretKeySource = process.env.OH_SECRET_KEY
    ? "custom (from OH_SECRET_KEY)"
    : "default (for local development)";

  console.log("Starting EXTRA standalone agent-server (shared state)...");
  console.log(`- agent-server: ${agentServerCmd.source}`);
  console.log(`- backend: ${config.backendBaseUrl}`);
  console.log(`- vscode port: ${config.vscodePort}`);
  console.log(`- shared state dir: ${config.stateDir}`);
  console.log(`- shared conversations: ${config.conversationsPath}`);
  console.log(`- secret key: ${secretKeySource}`);
  console.log("");
  console.log(
    "Connect via the GUI: open Add Backend, enter " +
      `${config.backendBaseUrl} as the host. Leave the API key blank ` +
      "unless this server is started with OH_SESSION_API_KEYS_0 set.",
  );
  console.log("");

  const backend = spawnProcess(
    agentServerCmd.command,
    [
      ...agentServerCmd.args,
      "--host",
      "127.0.0.1",
      "--port",
      String(config.backendPort),
    ],
    {
      cwd: config.cwd,
      env: {
        ...process.env,
        ...buildAgentServerEnv(config),
      },
    },
  );

  let shuttingDown = false;

  const shutdown = (signal = "SIGTERM") => {
    if (shuttingDown) {
      return;
    }

    shuttingDown = true;
    signalProcessTree(backend, signal);

    setTimeout(() => {
      if (isProcessRunning(backend)) {
        signalProcessTree(backend, "SIGKILL");
      }
      process.exit(process.exitCode ?? 0);
    }, 3000);
  };

  process.on("SIGINT", () => shutdown("SIGINT"));
  process.on("SIGTERM", () => shutdown("SIGTERM"));

  const backendErrored = new Promise((_, reject) => {
    backend.once("error", (error) => reject(error));
  });
  const backendExited = new Promise((_, reject) => {
    backend.once("exit", (code, signal) => {
      if (!shuttingDown) {
        reject(
          new Error(
            `agent-server exited before startup completed (code=${code ?? "null"}, signal=${signal ?? "null"})`,
          ),
        );
      }
    });
  });

  try {
    await Promise.race([
      waitForServer(`${config.backendBaseUrl}/server_info`),
      backendErrored,
      backendExited,
    ]);
  } catch (error) {
    shutdown();
    throw error;
  }

  console.log(`Extra agent-server is ready at ${config.backendBaseUrl}.`);

  backend.once("exit", (code) => {
    if (!shuttingDown) {
      console.error(
        `agent-server exited unexpectedly with code ${code ?? 0}`,
      );
      shutdown();
      process.exitCode = code ?? 1;
    } else {
      process.exitCode = code ?? 0;
    }
  });
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
  });
}
