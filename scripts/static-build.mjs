import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";

import {
  buildAutomationRuntimeServicesInfo,
  c,
  logError,
  logService,
  logStep,
  logSuccess,
} from "./dev-with-automation.mjs";
import { buildNpmScriptCommand } from "./dev-safe.mjs";

export function buildFrontend(config, args = {}) {
  const buildDir = join(config.canvasPath, "build");

  if (args.skipBuild) {
    if (!existsSync(buildDir)) {
      logError(
        "--skip-build was passed but build/ does not exist. Run without --skip-build first.",
      );
      process.exit(1);
    }
    logStep("build", "Skipping frontend build (--skip-build)");
    logService("build", `Reusing existing build/ at ${buildDir}`, c.dim);
    logService(
      "build",
      "Source edits will NOT appear until you run without --skip-build (or `npm run build`).",
      c.yellow,
    );
    return;
  }

  logStep("build", "Building frontend (npm run build:app)...");
  logService(
    "build",
    "This typically takes 30-60s; cached as build/ for --skip-build reuse",
    c.dim,
  );

  const cmd = buildNpmScriptCommand("build:app");
  const buildEnv = {
    ...process.env,
    // Bake the session API key — used by the frontend for both agent-server
    // and automation auth via the `X-Session-API-Key` header.
    VITE_SESSION_API_KEY: config.sessionApiKey,
    // Bake a description of the runtime services in this dev stack so the
    // frontend can populate the agent's <RUNTIME_SERVICES> system-prompt
    // block when creating a conversation.
    VITE_RUNTIME_SERVICES_INFO: JSON.stringify(
      buildAutomationRuntimeServicesInfo(config),
    ),
    // Intentionally do NOT set VITE_BACKEND_BASE_URL: leaving it unset makes
    // the runtime fall back to window.location.origin, which keeps the build
    // portable across localhost, LAN hosts, and tunnels such as ngrok.
  };
  if (config.viteWorkingDir) {
    buildEnv.VITE_WORKING_DIR = config.viteWorkingDir;
  }

  const result = spawnSync(cmd.command, cmd.args, {
    cwd: config.canvasPath,
    stdio: "inherit",
    env: buildEnv,
  });

  if (result.status !== 0) {
    logError(`Build failed with exit code ${result.status ?? "null"}`);
    process.exit(result.status ?? 1);
  }

  if (!existsSync(join(buildDir, "index.html"))) {
    logError(
      `Build completed but ${join(buildDir, "index.html")} is missing. ` +
        "Did react-router build write somewhere unexpected?",
    );
    process.exit(1);
  }

  logSuccess("Build complete");
}
