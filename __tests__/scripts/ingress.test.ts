import { createServer, type Server } from "node:http";
import { connect as netConnect, type AddressInfo, type Socket } from "node:net";
import type { Duplex } from "node:stream";
import { spawn, type ChildProcess } from "node:child_process";
import { once } from "node:events";
import path from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import { describe, expect, it, beforeAll, afterAll, afterEach } from "vitest";

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../..",
);

const ingressScript = path.join(repoRoot, "scripts", "ingress.mjs");
const loopbackHost = "127.0.0.1";

function originForPort(port: number) {
  return `http://${loopbackHost}:${port}`;
}

function serverPort(server: Server) {
  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("Expected server to be listening on a TCP port");
  }
  return (address as AddressInfo).port;
}

async function listenOnLoopback(server: Server) {
  await new Promise<void>((resolve, reject) => {
    const onError = (error: Error) => {
      server.off("error", onError);
      reject(error);
    };
    server.once("error", onError);
    server.listen(0, loopbackHost, () => {
      server.off("error", onError);
      resolve();
    });
  });
  return serverPort(server);
}

async function closeServer(server?: Server) {
  if (!server?.listening) {
    return;
  }
  await new Promise<void>((resolve, reject) => {
    server.close((error) => {
      if (error) {
        reject(error);
        return;
      }
      resolve();
    });
  });
}

async function getFreePort() {
  const server = createServer();
  try {
    return await listenOnLoopback(server);
  } finally {
    await closeServer(server);
  }
}

async function canConnect(port: number) {
  return new Promise<boolean>((resolve) => {
    const socket = netConnect({ host: loopbackHost, port });
    let settled = false;
    const finish = (connected: boolean) => {
      if (settled) {
        return;
      }
      settled = true;
      socket.destroy();
      resolve(connected);
    };
    socket.setTimeout(500);
    socket.once("connect", () => finish(true));
    socket.once("error", () => finish(false));
    socket.once("timeout", () => finish(false));
  });
}

async function waitForPort(port: number, child?: ChildProcess) {
  const deadline = Date.now() + 5000;
  while (Date.now() < deadline) {
    if (child && child.exitCode !== null) {
      throw new Error(
        `Process exited before port ${port} was ready: ${child.exitCode}`,
      );
    }
    if (await canConnect(port)) {
      return;
    }
    await delay(50);
  }
  throw new Error(`Timed out waiting for port ${port}`);
}

async function stopChild(child?: ChildProcess) {
  if (!child || child.exitCode !== null) {
    return;
  }
  child.kill("SIGTERM");
  const exited = once(child, "exit");
  const result = await Promise.race([
    exited.then(() => "exit" as const),
    delay(2000).then(() => "timeout" as const),
  ]);
  if (result === "timeout" && child.exitCode === null) {
    child.kill("SIGKILL");
    await Promise.race([exited, delay(1000)]);
  }
}

describe("ingress.mjs CLI", () => {
  it("shows help with --help flag", async () => {
    const child = spawn(process.execPath, [ingressScript, "--help"], {
      cwd: repoRoot,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let output = "";
    child.stdout.on("data", (chunk) => {
      output += chunk.toString();
    });

    const [code] = await once(child, "exit");

    expect(code).toBe(0);
    expect(output).toContain("Standalone Ingress / Reverse Proxy");
    expect(output).toContain("--port");
    expect(output).toContain("--route");
    expect(output).toContain("--default");
  });

  it("exits with error when no routes configured", async () => {
    const child = spawn(process.execPath, [ingressScript], {
      cwd: repoRoot,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stderr = "";
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });

    const [code] = await once(child, "exit");

    expect(code).toBe(1);
    expect(stderr).toContain("No routes configured");
  });

  it("parses --port argument correctly", async () => {
    const port = await getFreePort();
    const child = spawn(
      process.execPath,
      [
        ingressScript,
        "--port",
        port.toString(),
        "--default",
        "http://localhost:3000",
      ],
      {
        cwd: repoRoot,
        stdio: ["ignore", "pipe", "pipe"],
      },
    );

    let output = "";
    child.stdout.on("data", (chunk) => {
      output += chunk.toString();
    });

    await waitForPort(port, child);
    await stopChild(child);

    expect(output).toContain(port.toString());
  });

  it("parses --route arguments correctly", async () => {
    const port = await getFreePort();
    const child = spawn(
      process.execPath,
      [
        ingressScript,
        "--port",
        port.toString(),
        "--route",
        "/api=http://localhost:8000",
        "--route",
        "/static=http://localhost:3000",
      ],
      {
        cwd: repoRoot,
        stdio: ["ignore", "pipe", "pipe"],
      },
    );

    let output = "";
    child.stdout.on("data", (chunk) => {
      output += chunk.toString();
    });

    await waitForPort(port, child);
    await stopChild(child);

    expect(output).toContain("/api");
    expect(output).toContain("http://localhost:8000");
    expect(output).toContain("/static");
    expect(output).toContain("http://localhost:3000");
  });
});

describe("ingress proxy functionality", () => {
  let backend1: Server;
  let backend2: Server;
  let ingressProcess: ChildProcess;
  let backend1Port: number;
  let backend2Port: number;
  let ingressPort: number;

  beforeAll(async () => {
    // Create mock backend 1
    backend1 = createServer((req, res) => {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ backend: 1, path: req.url }));
    });
    backend1Port = await listenOnLoopback(backend1);

    // Create mock backend 2
    backend2 = createServer((req, res) => {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ backend: 2, path: req.url }));
    });
    backend2Port = await listenOnLoopback(backend2);

    // Start ingress
    ingressPort = await getFreePort();
    ingressProcess = spawn(
      process.execPath,
      [
        ingressScript,
        "--port",
        ingressPort.toString(),
        "--route",
        `/api/v2=${originForPort(backend2Port)}`,
        "--route",
        `/api=${originForPort(backend1Port)}`,
        "--default",
        originForPort(backend1Port),
      ],
      {
        cwd: repoRoot,
        stdio: ["ignore", "pipe", "pipe"],
      },
    );

    await waitForPort(ingressPort, ingressProcess);
  });

  afterAll(async () => {
    await stopChild(ingressProcess);
    await closeServer(backend1);
    await closeServer(backend2);
  });

  it("routes /api requests to backend1", async () => {
    const response = await fetch(`${originForPort(ingressPort)}/api/test`);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.backend).toBe(1);
    expect(data.path).toBe("/api/test");
  });

  it("routes /api/v2 requests to backend2 (more specific route)", async () => {
    const response = await fetch(`${originForPort(ingressPort)}/api/v2/test`);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.backend).toBe(2);
    expect(data.path).toBe("/api/v2/test");
  });

  it("routes unmatched paths to default backend", async () => {
    const response = await fetch(`${originForPort(ingressPort)}/other/path`);
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.backend).toBe(1);
    expect(data.path).toBe("/other/path");
  });

  it("preserves query parameters", async () => {
    const response = await fetch(
      `${originForPort(ingressPort)}/api/test?foo=bar&baz=123`,
    );
    const data = await response.json();

    expect(response.status).toBe(200);
    expect(data.path).toBe("/api/test?foo=bar&baz=123");
  });

  it("returns 502 when backend is unavailable", async () => {
    // Start a fresh ingress pointing to a non-existent backend
    const badBackendPort = await getFreePort();
    const badIngressPort = await getFreePort();
    const badIngress = spawn(
      process.execPath,
      [
        ingressScript,
        "--port",
        badIngressPort.toString(),
        "--default",
        originForPort(badBackendPort),
      ],
      {
        cwd: repoRoot,
        stdio: ["ignore", "pipe", "pipe"],
      },
    );

    await waitForPort(badIngressPort, badIngress);

    try {
      const response = await fetch(`${originForPort(badIngressPort)}/test`);
      expect(response.status).toBe(502);
      const text = await response.text();
      expect(text).toContain("Bad Gateway");
    } finally {
      await stopChild(badIngress);
    }
  });
});

describe("ingress route matching", () => {
  let backend: Server;
  let ingressProcess: ChildProcess;
  let backendPort: number;
  let ingressPort: number;

  beforeAll(async () => {
    backend = createServer((req, res) => {
      res.writeHead(200, { "Content-Type": "text/plain" });
      res.end(req.url);
    });
    backendPort = await listenOnLoopback(backend);

    ingressPort = await getFreePort();
    ingressProcess = spawn(
      process.execPath,
      [
        ingressScript,
        "--port",
        ingressPort.toString(),
        "--route",
        `/api/automation=${originForPort(backendPort)}`,
        "--route",
        `/api=${originForPort(backendPort)}`,
        "--route",
        `/sockets=${originForPort(backendPort)}`,
      ],
      {
        cwd: repoRoot,
        stdio: ["ignore", "pipe", "pipe"],
      },
    );

    await waitForPort(ingressPort, ingressProcess);
  });

  afterAll(async () => {
    await stopChild(ingressProcess);
    await closeServer(backend);
  });

  it("matches exact path", async () => {
    const response = await fetch(`${originForPort(ingressPort)}/api`);
    expect(response.status).toBe(200);
  });

  it("matches path with trailing content", async () => {
    const response = await fetch(`${originForPort(ingressPort)}/api/users`);
    expect(response.status).toBe(200);
    const text = await response.text();
    expect(text).toBe("/api/users");
  });

  it("matches longer prefix before shorter", async () => {
    const response = await fetch(
      `${originForPort(ingressPort)}/api/automation/docs`,
    );
    expect(response.status).toBe(200);
    const text = await response.text();
    expect(text).toBe("/api/automation/docs");
  });

  it("matches path with query string", async () => {
    const response = await fetch(`${originForPort(ingressPort)}/api?foo=bar`);
    expect(response.status).toBe(200);
    const text = await response.text();
    expect(text).toBe("/api?foo=bar");
  });

  it("returns 503 for unmatched routes with no default", async () => {
    const response = await fetch(`${originForPort(ingressPort)}/unknown`);
    expect(response.status).toBe(503);
  });
});

describe("ingress socket-error resilience", () => {
  // Regression coverage for crashes like:
  //   Error: read ECONNRESET ... Emitted 'error' event on Socket instance
  // which previously took down the whole ingress process when a WebSocket's
  // underlying TCP socket reset.
  let upstream: Server;
  let upstreamSockets: Duplex[];
  let ingressProcess: ChildProcess;
  let ingressStderr: string;
  let upstreamPort: number;
  let ingressPort: number;

  beforeAll(async () => {
    upstreamSockets = [];

    upstream = createServer((req, res) => {
      res.writeHead(200, { "Content-Type": "text/plain" });
      res.end("ok");
    });

    // Accept WebSocket upgrades, immediately RST the upstream socket on the
    // next tick. This reproduces the production crash without requiring a
    // real WebSocket handshake handler.
    upstream.on("upgrade", (_req, socket) => {
      upstreamSockets.push(socket);
      socket.write(
        "HTTP/1.1 101 Switching Protocols\r\n" +
          "Upgrade: websocket\r\n" +
          "Connection: Upgrade\r\n\r\n",
      );
      // Force a TCP RST instead of a clean FIN to mirror ECONNRESET.
      setImmediate(() => {
        const s = socket as Duplex & { resetAndDestroy?: () => void };
        if (typeof s.resetAndDestroy === "function") {
          s.resetAndDestroy();
        } else {
          s.destroy(
            Object.assign(new Error("forced reset"), { code: "ECONNRESET" }),
          );
        }
      });
    });

    upstreamPort = await listenOnLoopback(upstream);

    ingressPort = await getFreePort();
    ingressStderr = "";
    ingressProcess = spawn(
      process.execPath,
      [
        ingressScript,
        "--port",
        ingressPort.toString(),
        "--default",
        originForPort(upstreamPort),
      ],
      {
        cwd: repoRoot,
        stdio: ["ignore", "pipe", "pipe"],
      },
    );
    ingressProcess.stderr?.on("data", (chunk) => {
      ingressStderr += chunk.toString();
    });

    await waitForPort(ingressPort, ingressProcess);
  });

  afterAll(async () => {
    await stopChild(ingressProcess);
    for (const s of upstreamSockets) {
      try {
        s.destroy();
      } catch {
        // ignore
      }
    }
    await closeServer(upstream);
  });

  function openWebSocketHandshake(port: number): Promise<Socket> {
    return new Promise((resolve, reject) => {
      const sock = netConnect({ host: "127.0.0.1", port }, () => {
        sock.write(
          "GET /sockets/events/test HTTP/1.1\r\n" +
            `Host: 127.0.0.1:${port}\r\n` +
            "Upgrade: websocket\r\n" +
            "Connection: Upgrade\r\n" +
            "Sec-WebSocket-Version: 13\r\n" +
            "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n",
        );
        resolve(sock);
      });
      // The upstream RST will surface here as ECONNRESET; we just want the
      // handshake to be initiated, so swallow any error after that.
      sock.on("error", () => {});
      sock.once("error", reject);
    });
  }

  it("survives upstream WebSocket ECONNRESET without crashing", async () => {
    // Trigger the bug repeatedly to make sure no path crashes the proxy.
    for (let i = 0; i < 5; i++) {
      const client = await openWebSocketHandshake(ingressPort);
      // Wait long enough for the upstream RST to propagate through the proxy.
      await delay(150);
      client.destroy();
      await delay(50);
    }

    // Process must still be alive.
    expect(ingressProcess.exitCode).toBeNull();
    expect(ingressProcess.signalCode).toBeNull();

    // And it must still be serving HTTP traffic.
    const response = await fetch(`${originForPort(ingressPort)}/health`);
    expect(response.status).toBe(200);

    // The unhandled-error crash signature must not appear in stderr.
    expect(ingressStderr).not.toContain("Unhandled 'error' event");
    expect(ingressStderr).not.toMatch(/throw er;/);
  });

  it("survives client aborting an in-flight HTTP request", async () => {
    // Open and abruptly destroy a TCP connection mid-request to make sure
    // req/res 'error' events on the client side are handled.
    const client = netConnect({ host: "127.0.0.1", port: ingressPort }, () => {
      client.write(
        "GET /something HTTP/1.1\r\n" +
          `Host: 127.0.0.1:${ingressPort}\r\n` +
          "Connection: close\r\n\r\n",
      );
      // Reset before the upstream finishes responding.
      setImmediate(() => client.destroy());
    });
    client.on("error", () => {});

    await delay(200);

    expect(ingressProcess.exitCode).toBeNull();
    expect(ingressProcess.signalCode).toBeNull();
    expect(ingressStderr).not.toContain("Unhandled 'error' event");
  });
});
