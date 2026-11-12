import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { createServer, request, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { vi } from "vitest";
import { WebSocketServer } from "ws";

import { startStaticServer } from "../../scripts/static-server.mjs";
import type { startPortForwardTunnel } from "../../scripts/tunnel-client.mjs";

const KEY_HEADER = "x-session-api-key";

function listen(server: Server): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject);
      resolve((server.address() as AddressInfo).port);
    });
  });
}

function close(server: Server): Promise<void> {
  server.closeAllConnections?.();
  return new Promise((resolve) => {
    server.close(() => resolve());
  });
}

/**
 * A DigitalOcean session's guest port as the `openhands` template serves it
 * since November 2026: an agent-server that wants its own session key, behind
 * Agent Canvas's real static-server, which writes that key into its page.
 * `key: null` is a guest without a key. `keysSeen` records the key header of
 * every request and upgrade that reached the agent-server.
 */
export async function startKeyedGuest(key: string | null) {
  const keysSeen: (string | undefined)[] = [];
  const presented = (headers: Record<string, unknown>) =>
    headers[KEY_HEADER] as string | undefined;

  const agentServer = createServer((req, res) => {
    keysSeen.push(presented(req.headers));
    if (key && presented(req.headers) !== key) {
      res.writeHead(401, { "content-type": "application/json" });
      res.end('{"detail":"Unauthorized"}');
      return;
    }
    res.writeHead(200, { "content-type": "application/json" });
    res.end(JSON.stringify({ path: req.url }));
  });
  const sockets = new WebSocketServer({ noServer: true });
  agentServer.on("upgrade", (req, socket, head) => {
    keysSeen.push(presented(req.headers));
    if (key && presented(req.headers) !== key) {
      socket.end("HTTP/1.1 401 Unauthorized\r\n\r\n");
      return;
    }
    sockets.handleUpgrade(req, socket, head, (ws) => {
      ws.on("message", (data) => ws.send(`echo:${String(data)}`));
    });
  });
  const agentUrl = `http://127.0.0.1:${await listen(agentServer)}`;

  const dir = mkdtempSync(join(tmpdir(), "keyed-guest-"));
  writeFileSync(
    join(dir, "index.html"),
    "<html><head></head><body>canvas</body></html>",
  );
  const front: Server = await startStaticServer({
    port: 0,
    host: "127.0.0.1",
    dir,
    routes: {
      "/api": agentUrl,
      "/sockets": agentUrl,
      "/server_info": agentUrl,
      "/alive": agentUrl,
    },
    sessionApiKey: key,
  });

  return {
    port: (front.address() as AddressInfo).port,
    keysSeen,
    async close() {
      for (const client of sockets.clients) client.terminate();
      await Promise.all([close(front), close(agentServer)]);
      rmSync(dir, { recursive: true, force: true });
    },
  };
}

/**
 * The status of a GET through node:http. These tests call agent-server paths,
 * where the repo's lint rule keeps global fetch out.
 */
export function statusOf(
  url: string,
  headers: Record<string, string> = {},
): Promise<number> {
  return new Promise((resolve, reject) => {
    request(url, { headers }, (response) => {
      response.resume();
      resolve(response.statusCode ?? 0);
    })
      .on("error", reject)
      .end();
  });
}

/**
 * Stands in for startPortForwardTunnel without dialing DigitalOcean: the raw
 * tunnel's port is `localPort`, a guest started here.
 */
export function standInTunnel(localPort: number) {
  const stop = vi.fn();
  const startTunnel = vi.fn<typeof startPortForwardTunnel>(async (options) => ({
    sessionId: options?.sessionId ?? "",
    remotePort: options?.remotePort ?? 0,
    localPort,
    getLastUpstreamFailure: () => null,
    stop,
  }));
  return { startTunnel, stop };
}
