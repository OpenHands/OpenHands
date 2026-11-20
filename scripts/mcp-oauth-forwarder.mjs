#!/usr/bin/env node
/**
 * MCP OAuth callback forwarder for a Dockerised Agent Canvas.
 *
 * Problem: when you install an OAuth MCP server (e.g. GitLab), the Agent
 * Server inside the container opens a one-off callback listener on a random
 * port (log line: "OAuth callback server started on http://localhost:48751").
 * The OAuth provider then redirects your browser to that address, but the
 * port only exists inside the container, so the browser shows
 * "localhost refused to connect".
 *
 * Fix: watch the container log. Each time a callback listener starts, listen
 * on the same port on this machine and relay every connection into the
 * container (through `docker exec`), so the browser redirect just works.
 *
 * Usage:
 *   node scripts/mcp-oauth-forwarder.mjs [container-name]   (default: agent-canvas)
 *
 * It exits by itself when the container stops. Nothing is stored; the OAuth
 * code only passes through to the container, which is where it is meant to go.
 */
import { spawn } from "node:child_process";
import net from "node:net";

const container = process.argv[2] || process.env.CONTAINER_NAME || "agent-canvas";
const LISTEN_MS = 10 * 60 * 1000; // close an unused forwarding port after 10 min

// Runs inside the container: copy bytes between stdin/stdout and the port.
const RELAY_PY = `
import socket, sys, threading
s = socket.create_connection(("localhost", int(sys.argv[1])))
def up():
    try:
        while True:
            d = sys.stdin.buffer.read1(65536)
            if not d: break
            s.sendall(d)
    except Exception: pass
    try: s.shutdown(socket.SHUT_WR)
    except Exception: pass
threading.Thread(target=up, daemon=True).start()
try:
    while True:
        d = s.recv(65536)
        if not d: break
        sys.stdout.buffer.write(d); sys.stdout.buffer.flush()
except Exception: pass
`;

const log = (msg) => console.log(`[mcp-oauth-forwarder] ${msg}`);
const servers = new Map(); // port -> net.Server

function forward(port) {
  if (servers.has(port)) return;
  const server = net.createServer((client) => {
    const relay = spawn(
      "docker",
      ["exec", "-i", container, "python3", "-c", RELAY_PY, String(port)],
      { stdio: ["pipe", "pipe", "ignore"] },
    );
    client.pipe(relay.stdin);
    relay.stdout.pipe(client);
    const done = () => {
      client.destroy();
      relay.kill();
    };
    client.on("close", done);
    client.on("error", done);
    relay.on("close", () => client.end());
    relay.stdin.on("error", () => {});
  });
  server.on("error", (err) =>
    log(`cannot listen on 127.0.0.1:${port}: ${err.message}`),
  );
  server.listen(port, "127.0.0.1", () => {
    servers.set(port, server);
    log(`forwarding localhost:${port} -> container '${container}':${port}`);
    setTimeout(() => {
      server.close();
      servers.delete(port);
      log(`closed forwarding port ${port}`);
    }, LISTEN_MS).unref();
  });
}

// Only look at new log lines: --since now avoids replaying old listeners.
const logs = spawn(
  "docker",
  ["logs", "-f", "--since", new Date().toISOString(), container],
  { stdio: ["ignore", "pipe", "pipe"] },
);

// The log wraps the message ("...started on" / "http://localhost:PORT"), so
// keep a short rolling window and search that.
let window = "";
const onData = (chunk) => {
  window = (window + chunk.toString()).slice(-600);
  // The logger puts its source ("oauth.py:440") between the message and the
  // wrapped URL, so allow some text in between.
  const m = window.match(
    /callback server started on[\s\S]{0,160}?http:\/\/localhost:(\d+)/i,
  );
  if (m) {
    window = "";
    forward(Number(m[1]));
  }
};
logs.stdout.on("data", onData);
logs.stderr.on("data", onData);
logs.on("close", () => {
  log(`container '${container}' log ended; exiting`);
  for (const s of servers.values()) s.close();
  process.exit(0);
});
logs.on("error", (err) => {
  console.error(`docker not available: ${err.message}`);
  process.exit(1);
});
log(`watching '${container}' for OAuth callback listeners (Ctrl+C to stop)`);
