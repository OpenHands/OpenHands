import { expect, test } from "@playwright/test";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import type { Server } from "node:http";
import net from "node:net";
import { networkInterfaces, tmpdir } from "node:os";
import path from "node:path";

import { startStaticServer } from "../../../scripts/static-server.mjs";

const ASSET_DIR = path.join(process.cwd(), ".github", "pr-assets");

function originOf(server: Server): string {
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("Static server did not bind to a TCP port");
  }
  return `http://127.0.0.1:${address.port}`;
}

function closeServer(server: Server): Promise<void> {
  return new Promise((resolve) => {
    server.close(() => resolve());
  });
}

function firstLanIpv4(): string | null {
  for (const addrs of Object.values(networkInterfaces())) {
    for (const addr of addrs ?? []) {
      if (addr.family === "IPv4" && !addr.internal) {
        return addr.address;
      }
    }
  }
  return null;
}

function canConnect(host: string, port: number, ms = 1500): Promise<boolean> {
  return new Promise((resolve) => {
    const socket = net.connect({ host, port }, () => {
      socket.destroy();
      resolve(true);
    });
    socket.setTimeout(ms, () => {
      socket.destroy();
      resolve(false);
    });
    socket.on("error", () => resolve(false));
  });
}

function escapeHtml(text: string): string {
  return text
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

test("LAN bind strips the session key from GET / and writes a PNG", async ({
  page,
}) => {
  const dir = mkdtempSync(path.join(tmpdir(), "oh-bind-"));
  writeFileSync(
    path.join(dir, "index.html"),
    "<html><head></head><body>app</body></html>",
  );
  const servers: Server[] = [];

  try {
    const loopback = await startStaticServer({
      port: 0,
      host: "127.0.0.1",
      dir,
      routes: {},
      sessionApiKey: "loopback-fixture-key",
    });
    servers.push(loopback);

    const lan = await startStaticServer({
      port: 0,
      host: "0.0.0.0",
      dir,
      routes: {},
      sessionApiKey: "lan-secret",
    });
    servers.push(lan);

    const loopbackHtml = await (await fetch(`${originOf(loopback)}/`)).text();
    const lanHtml = await (await fetch(`${originOf(lan)}/`)).text();

    expect(loopbackHtml).toContain("__AGENT_CANVAS_SESSION_API_KEY__");
    expect(loopbackHtml).toContain("loopback-fixture-key");
    expect(lanHtml).not.toContain("lan-secret");
    expect(lanHtml).not.toContain("__AGENT_CANVAS_SESSION_API_KEY__");
    expect(lanHtml).toContain("__AGENT_CANVAS_AUTH_REQUIRED__");

    mkdirSync(ASSET_DIR, { recursive: true });
    await page.setContent(`<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>OpenHands #16879 bind policy</title>
  <style>
    body { margin: 0; background: #0d1117; color: #c9d1d9; font: 14px/1.45 ui-monospace, Consolas, monospace; padding: 24px; }
    h1 { font-size: 16px; color: #8b949e; font-weight: 600; }
    h2 { font-size: 15px; margin: 20px 0 8px; }
    .before { color: #f85149; }
    .after { color: #3fb950; }
    pre { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 16px; white-space: pre-wrap; word-break: break-all; }
  </style>
</head>
<body>
  <h1>OpenHands #16879 — GET / index.html (Playwright)</h1>
  <h2 class="before">BEFORE pattern: host 0.0.0.0 + session key (would leak on LAN)</h2>
  <pre>window.__AGENT_CANVAS_SESSION_API_KEY__="&lt;redacted&gt;"</pre>
  <h2 class="after">AFTER this branch: host 0.0.0.0 without --allow-lan-session-key</h2>
  <pre>${escapeHtml(lanHtml.replaceAll("lan-secret", "<redacted>"))}</pre>
  <h2>LOOPBACK still injects for local use</h2>
  <pre>${escapeHtml(loopbackHtml.replaceAll("loopback-fixture-key", "<redacted>"))}</pre>
</body>
</html>`);
    await page.screenshot({
      path: path.join(ASSET_DIR, "pr-16975-bind-policy.png"),
      fullPage: true,
    });
  } finally {
    await Promise.all(servers.map(closeServer));
    rmSync(dir, { recursive: true, force: true });
  }
});

test("live ingress is loopback-only when the stack is already up", async ({
  page,
}) => {
  const ingressUrl = process.env.OH_INGRESS_URL ?? "http://127.0.0.1:8000/";
  let reachable = false;
  try {
    const response = await fetch(ingressUrl, { signal: AbortSignal.timeout(3000) });
    reachable = response.ok;
  } catch {
    reachable = false;
  }
  test.skip(!reachable, `Start the stack first: node scripts/dev-with-automation.mjs --port 8000`);

  const loopbackOpen = await canConnect("127.0.0.1", 8000);
  expect(loopbackOpen).toBe(true);

  const lan = firstLanIpv4();
  if (lan) {
    const lanOpen = await canConnect(lan, 8000);
    expect(lanOpen, `ingress should refuse ${lan}:8000`).toBe(false);
  }

  await page.goto(ingressUrl, { waitUntil: "domcontentloaded" });
  await expect(page.getByRole("heading", { name: "Choose your agent" })).toBeVisible({
    timeout: 30_000,
  });
  mkdirSync(ASSET_DIR, { recursive: true });
  await page.screenshot({
    path: path.join(ASSET_DIR, "pr-16975-agent-canvas-loopback.png"),
    fullPage: false,
  });
});
