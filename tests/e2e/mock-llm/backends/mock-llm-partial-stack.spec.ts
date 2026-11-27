/**
 * Mock-LLM E2E tests for partial stack modes (--frontend-only, --backend-only)
 * and port conflict handling.
 *
 * These tests spawn `bin/agent-canvas.mjs` with partial-stack flags and verify:
 *   1. --frontend-only: static frontend is served, backend APIs return 503
 *   2. --backend-only: backend APIs work, frontend root returns 503
 *   3. Port conflict: binary fails with a clear error when the port is busy,
 *      then succeeds when a free port is used
 *
 * Unlike the other mock-llm specs these tests do NOT rely on the webServer
 * entries in playwright.mock-llm.config.ts — they manage their own child
 * processes so each test controls exactly which flags are passed.
 */

import { test, expect } from "@playwright/test";
import { spawn, type ChildProcess } from "node:child_process";
import { randomBytes } from "node:crypto";
import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { createServer, type Server } from "node:net";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  FRONTEND_ONLY_INGRESS_PORT,
  FRONTEND_ONLY_URL,
  BACKEND_ONLY_INGRESS_PORT,
  BACKEND_ONLY_URL,
  seedLocalStorage,
} from "../utils/mock-llm-helpers";

// ── Paths ──────────────────────────────────────────────────────────────
const PROJECT_ROOT = resolve(
  dirname(fileURLToPath(import.meta.url)),
  "../../../..",
);
const BIN = join(PROJECT_ROOT, "bin/agent-canvas.mjs");

// ── Helpers ────────────────────────────────────────────────────────────

/** Spawn `bin/agent-canvas.mjs` with the given CLI flags and env overrides. */
function spawnAgentCanvas(
  flags: string[],
  env: Record<string, string> = {},
): ChildProcess {
  const child = spawn("node", [BIN, ...flags], {
    cwd: PROJECT_ROOT,
    env: {
      ...process.env,
      // Suppress analytics in child processes
      VITE_DO_NOT_TRACK: "1",
      VITE_ENABLE_BROWSER_TOOLS: "false",
      ...env,
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  return child;
}

/** Collect stdout + stderr from a child process into a buffer. */
function collectOutput(child: ChildProcess): { get(): string } {
  let buf = "";
  child.stdout?.on("data", (chunk: Buffer) => {
    buf += chunk.toString();
  });
  child.stderr?.on("data", (chunk: Buffer) => {
    buf += chunk.toString();
  });
  return { get: () => buf };
}

/** Wait for a child process to exit within a timeout. */
async function waitForExit(
  child: ChildProcess,
  timeoutMs = 15_000,
): Promise<{ code: number | null; signal: string | null; timedOut: boolean }> {
  return new Promise((resolve) => {
    const timer = setTimeout(() => {
      resolve({ code: null, signal: null, timedOut: true });
    }, timeoutMs);

    child.on("exit", (code, signal) => {
      clearTimeout(timer);
      resolve({ code, signal, timedOut: false });
    });
  });
}

/**
 * Poll a URL until it returns a non-5xx response or timeout.
 * Returns the HTTP status or null if the service never became ready.
 */
async function pollUrl(
  url: string,
  timeoutMs = 120_000,
  intervalMs = 1_000,
): Promise<number | null> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const resp = await fetch(url, {
        signal: AbortSignal.timeout(2_000),
      });
      // Treat 502/503/504 as "not ready yet" — the ingress can proxy but
      // the upstream service hasn't started. Only return on a real response.
      if (resp.status < 500) return resp.status;
    } catch {
      // Connection refused — service not ready yet
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  return null;
}

/** Create an isolated state dir + session key for a child stack instance. */
function createIsolatedEnv(
  ingressPort: string,
  extra: Record<string, string> = {},
): { env: Record<string, string>; stateDir: string } {
  const stateDir = mkdtempSync(join(tmpdir(), "partial-stack-"));
  const sessionKey = randomBytes(32).toString("hex");
  return {
    stateDir,
    env: {
      OH_CANVAS_SAFE_STATE_DIR: stateDir,
      PORT: ingressPort,
      LOCAL_BACKEND_API_KEY: sessionKey,
      // Use isolated ports for backend services (high ports unlikely to collide)
      OH_CANVAS_SAFE_BACKEND_PORT: String(parseInt(ingressPort) + 1),
      OH_CANVAS_SAFE_AUTOMATION_PORT: String(parseInt(ingressPort) + 2),
      OH_CANVAS_SAFE_VITE_PORT: String(parseInt(ingressPort) + 3),
      // Isolated key file so we don't touch ~/.openhands
      OH_SESSION_API_KEY_PATH: join(stateDir, "session-key.txt"),
      ...extra,
    },
  };
}

/** Gracefully kill a child process and wait for it to exit. */
async function killChild(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null) return; // already exited
  child.kill("SIGTERM");
  await new Promise<void>((resolve) => {
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      resolve();
    }, 5_000);
    child.on("exit", () => {
      clearTimeout(timer);
      resolve();
    });
  });
}

/** Block a TCP port and return the server handle. */
function blockPort(port: number): Promise<Server> {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.listen(port, "127.0.0.1", () => resolve(server));
    server.on("error", reject);
  });
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

test.describe.configure({ mode: "serial" });

// ───────────────────────────────────────────────────────────────────────
// 1. Frontend-only mode
// ───────────────────────────────────────────────────────────────────────

test.describe("partial stack: --frontend-only", () => {
  let child: ChildProcess;
  let stateDir: string;

  test.afterEach(async () => {
    if (child) await killChild(child);
    if (stateDir) rmSync(stateDir, { recursive: true, force: true });
  });

  test("serves the frontend but returns 503 for backend routes", async ({
    page,
    request,
  }) => {
    test.setTimeout(60_000);

    // These tests spawn bin/agent-canvas.mjs locally, which needs a
    // pre-built frontend. Skip when running in Docker-only CI (no local build).
    test.skip(
      !existsSync(join(PROJECT_ROOT, "build/index.html")),
      "build/index.html missing — skipped (run `npm run build:app` or use the npm e2e config)",
    );

    const isolated = createIsolatedEnv(FRONTEND_ONLY_INGRESS_PORT);
    stateDir = isolated.stateDir;
    const output = collectOutput(
      (child = spawnAgentCanvas(["--frontend-only"], isolated.env)),
    );

    // Wait for the ingress to start serving
    const rootStatus = await pollUrl(`${FRONTEND_ONLY_URL}/`, 60_000);
    expect(
      rootStatus,
      `Frontend-only ingress never became ready.\nOutput: ${output.get().slice(-500)}`,
    ).toBe(200);

    // Verify: root returns HTML (the static frontend)
    const rootResp = await request.get(`${FRONTEND_ONLY_URL}/`);
    expect(rootResp.status()).toBe(200);
    const html = await rootResp.text();
    expect(html).toContain("<!DOCTYPE html");

    // Verify: backend API routes return 503 because the static server
    // rejects known API prefixes via --reject-prefix when no backend
    // is configured (frontend-only mode).
    const serverInfoResp = await request.get(
      `${FRONTEND_ONLY_URL}/server_info`,
      { failOnStatusCode: false },
    );
    expect(serverInfoResp.status()).toBe(503);

    const settingsResp = await request.get(
      `${FRONTEND_ONLY_URL}/api/settings`,
      { failOnStatusCode: false },
    );
    expect(settingsResp.status()).toBe(503);

    const automationResp = await request.get(
      `${FRONTEND_ONLY_URL}/api/automation/v1`,
      { failOnStatusCode: false },
    );
    expect(automationResp.status()).toBe(503);

    // Verify: browser loads the app and shows "agent server unavailable"
    // because the /server_info probe fails with 503.
    await seedLocalStorage(page);
    await page.goto(FRONTEND_ONLY_URL, { waitUntil: "domcontentloaded" });

    // The app should detect the missing backend and show the manage-backends
    // modal or an equivalent unavailable notice.
    await expect(
      page.getByTestId("manage-backends-modal"),
    ).toBeVisible({ timeout: 15_000 });
  });
});

// ───────────────────────────────────────────────────────────────────────
// 2. Backend-only mode
// ───────────────────────────────────────────────────────────────────────

test.describe("partial stack: --backend-only", () => {
  let child: ChildProcess;
  let stateDir: string;

  test.afterEach(async () => {
    if (child) await killChild(child);
    if (stateDir) rmSync(stateDir, { recursive: true, force: true });
  });

  test("serves backend APIs but returns 503 for the frontend root", async ({
    request,
  }) => {
    // Backend-only needs uvx → agent-server + automation startup: allow 3 min
    test.setTimeout(180_000);

    const isolated = createIsolatedEnv(BACKEND_ONLY_INGRESS_PORT);
    stateDir = isolated.stateDir;
    const sessionKey = isolated.env.LOCAL_BACKEND_API_KEY;
    const output = collectOutput(
      (child = spawnAgentCanvas(["--backend-only"], isolated.env)),
    );

    // Wait for agent-server to be ready through the ingress
    const serverInfoStatus = await pollUrl(
      `${BACKEND_ONLY_URL}/server_info`,
      150_000,
    );
    expect(
      serverInfoStatus,
      `Backend-only agent-server never became ready.\nOutput: ${output.get().slice(-800)}`,
    ).not.toBeNull();

    // Wait for automation backend to also be ready (starts independently,
    // may lag behind the agent-server).
    const automationStatus = await pollUrl(
      `${BACKEND_ONLY_URL}/api/automation/v1`,
      60_000,
    );
    expect(
      automationStatus,
      `Backend-only automation never became ready.\nOutput: ${output.get().slice(-800)}`,
    ).not.toBeNull();

    // Verify: /server_info returns 200 (agent-server running)
    const serverInfoResp = await request.get(
      `${BACKEND_ONLY_URL}/server_info`,
    );
    expect(serverInfoResp.status()).toBe(200);
    const serverInfo = await serverInfoResp.json();
    expect(serverInfo).toHaveProperty("version");

    // Verify: /api/settings is reachable (may return 401 without key, but not 503)
    const settingsResp = await request.get(
      `${BACKEND_ONLY_URL}/api/settings`,
      {
        headers: { "X-Session-API-Key": sessionKey },
        failOnStatusCode: false,
      },
    );
    expect(settingsResp.status()).not.toBe(503);

    // Verify: automation backend is also running and reachable
    const automationResp = await request.get(
      `${BACKEND_ONLY_URL}/api/automation/v1`,
      { failOnStatusCode: false },
    );
    expect([200, 401]).toContain(automationResp.status());

    // Verify: frontend root returns 503 (no default backend in ingress)
    const rootResp = await request.get(`${BACKEND_ONLY_URL}/`, {
      failOnStatusCode: false,
    });
    expect(rootResp.status()).toBe(503);

    // Verify: a random static-asset-like path also returns 503
    const assetResp = await request.get(
      `${BACKEND_ONLY_URL}/assets/index.js`,
      { failOnStatusCode: false },
    );
    expect(assetResp.status()).toBe(503);
  });
});

// ───────────────────────────────────────────────────────────────────────
// 3. Port conflict handling
// ───────────────────────────────────────────────────────────────────────

test.describe("partial stack: port conflict", () => {
  let child: ChildProcess;
  let blocker: Server | null = null;
  let stateDir: string;

  test.afterEach(async () => {
    if (child) await killChild(child);
    if (blocker) {
      await new Promise<void>((r) => blocker!.close(() => r()));
      blocker = null;
    }
    if (stateDir) rmSync(stateDir, { recursive: true, force: true });
  });

  test("fails with a clear error when the ingress port is occupied", async () => {
    test.setTimeout(30_000);

    test.skip(
      !existsSync(join(PROJECT_ROOT, "build/index.html")),
      "build/index.html missing — skipped (run `npm run build:app` or use the npm e2e config)",
    );

    const conflictPort = 18330;

    // Block the port with a dummy TCP server
    blocker = await blockPort(conflictPort);

    const isolated = createIsolatedEnv(String(conflictPort));
    stateDir = isolated.stateDir;
    const output = collectOutput(
      (child = spawnAgentCanvas(["--frontend-only"], isolated.env)),
    );

    // The process should exit non-zero because the port is busy
    const result = await waitForExit(child, 20_000);

    expect(result.timedOut, "Process should exit promptly on port conflict").toBe(
      false,
    );
    expect(result.code, "Process should exit non-zero on port conflict").not.toBe(
      0,
    );

    // The error message should mention the blocked port
    const text = output.get();
    expect(text).toMatch(
      new RegExp(`(port\\s+${conflictPort}|${conflictPort}.*in use|EADDRINUSE)`, "i"),
    );
  });

  test("starts successfully on a free port after a conflict", async ({
    request,
  }) => {
    test.setTimeout(60_000);

    // Use a port that is NOT blocked — frontend-only starts fast (no uvx)
    const freePort = 18331;

    test.skip(
      !existsSync(join(PROJECT_ROOT, "build/index.html")),
      "build/index.html missing — skipped (run `npm run build:app` or use the npm e2e config)",
    );

    const isolated = createIsolatedEnv(String(freePort));
    stateDir = isolated.stateDir;
    const output = collectOutput(
      (child = spawnAgentCanvas(["--frontend-only"], isolated.env)),
    );

    const freeUrl = `http://localhost:${freePort}`;
    const rootStatus = await pollUrl(`${freeUrl}/`, 30_000);
    expect(
      rootStatus,
      `Frontend-only on free port never became ready.\nOutput: ${output.get().slice(-500)}`,
    ).toBe(200);

    // Verify it actually serves content
    const rootResp = await request.get(`${freeUrl}/`);
    expect(rootResp.status()).toBe(200);
    const html = await rootResp.text();
    expect(html).toContain("<!DOCTYPE html");
  });
});
