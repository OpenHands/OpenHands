// @vitest-environment node
import { createServer, request as httpRequest } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it } from "vitest";

import {
  isAllowedCloudProxyHost,
  parseCloudProxyEnvelope,
  handleCloudProxy,
  CLOUD_PROXY_PATH,
} from "../../scripts/cloud-proxy.mjs";

describe("cloud-proxy allowlist", () => {
  it.each([
    ["abc.prod-runtime.all-hands.dev", true],
    ["x.staging-runtime.all-hands.dev", true],
    ["foo.runtime.all-hands.dev", true],
    ["localhost", true],
    ["127.0.0.1", true],
    ["app.all-hands.dev", false],
    ["evil.example.com", false],
    ["", false],
  ])("isAllowedCloudProxyHost(%s) → %s", (host, allowed) => {
    expect(isAllowedCloudProxyHost(host)).toBe(allowed);
  });
});

describe("parseCloudProxyEnvelope", () => {
  it("accepts a runtime sandbox envelope", () => {
    const parsed = parseCloudProxyEnvelope({
      host: "https://abc.prod-runtime.all-hands.dev",
      method: "POST",
      path: "/api/bash/execute_bash_command",
      headers: { "X-Session-API-Key": "k" },
      body: { command: "ls" },
      timeout_seconds: 45,
    });

    expect(parsed.method).toBe("POST");
    expect(parsed.upstreamUrl.href).toBe(
      "https://abc.prod-runtime.all-hands.dev/api/bash/execute_bash_command",
    );
    expect(parsed.timeoutMs).toBe(45_000);
    expect(parsed.headers).toMatchObject({ "X-Session-API-Key": "k" });
  });

  it("rejects non-allowlisted hosts", () => {
    expect(() =>
      parseCloudProxyEnvelope({
        host: "https://evil.example.com",
        method: "GET",
        path: "/secret",
      }),
    ).toThrow(/allowlisted/);
  });

  it("rejects http for non-loopback hosts", () => {
    expect(() =>
      parseCloudProxyEnvelope({
        host: "http://abc.prod-runtime.all-hands.dev",
        method: "GET",
        path: "/api/health",
      }),
    ).toThrow(/https/);
  });
});

describe("handleCloudProxy", () => {
  const servers: ReturnType<typeof createServer>[] = [];

  afterEach(async () => {
    await Promise.all(
      servers.splice(0).map(
        (server) =>
          new Promise<void>((resolve, reject) => {
            server.close((error) => (error ? reject(error) : resolve()));
          }),
      ),
    );
  });

  it("forwards the envelope to an allowlisted loopback upstream", async () => {
    const upstreamHits: Array<{ method?: string; url?: string; body: string }> =
      [];
    const upstream = createServer((req, res) => {
      const chunks: Buffer[] = [];
      req.on("data", (chunk) => chunks.push(chunk));
      req.on("end", () => {
        upstreamHits.push({
          method: req.method,
          url: req.url,
          body: Buffer.concat(chunks).toString("utf8"),
        });
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ exit_code: 0, stdout: "./a\n" }));
      });
    });
    servers.push(upstream);
    await listen(upstream);
    const upstreamPort = (upstream.address() as AddressInfo).port;

    const proxy = createServer((req, res) => {
      void handleCloudProxy(req, res);
    });
    servers.push(proxy);
    await listen(proxy);
    const proxyPort = (proxy.address() as AddressInfo).port;

    // Use node:http (not fetch) so MSW in vitest.setup cannot intercept.
    const response = await postJson(proxyPort, {
      host: `http://127.0.0.1:${upstreamPort}`,
      method: "POST",
      path: "/api/bash/execute_bash_command",
      headers: { "X-Session-API-Key": "sess" },
      body: { command: "find ." },
    });

    expect(response.statusCode).toBe(200);
    expect(JSON.parse(response.body)).toEqual({
      exit_code: 0,
      stdout: "./a\n",
    });
    expect(upstreamHits).toHaveLength(1);
    expect(upstreamHits[0]).toMatchObject({
      method: "POST",
      url: "/api/bash/execute_bash_command",
    });
    expect(JSON.parse(upstreamHits[0]!.body)).toEqual({ command: "find ." });
  });

  it("returns 403 for disallowed upstream hosts", async () => {
    const proxy = createServer((req, res) => {
      void handleCloudProxy(req, res);
    });
    servers.push(proxy);
    await listen(proxy);
    const proxyPort = (proxy.address() as AddressInfo).port;

    const response = await postJson(proxyPort, {
      host: "https://evil.example.com",
      method: "GET",
      path: "/secret",
    });

    expect(response.statusCode).toBe(403);
    expect(JSON.parse(response.body)).toMatchObject({
      detail: expect.stringMatching(/allowlisted/),
    });
  });
});

function listen(server: ReturnType<typeof createServer>) {
  return new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => resolve());
  });
}

function postJson(
  port: number,
  envelope: unknown,
): Promise<{ statusCode: number; body: string }> {
  const payload = JSON.stringify(envelope);
  return new Promise((resolve, reject) => {
    const req = httpRequest(
      {
        hostname: "127.0.0.1",
        port,
        path: CLOUD_PROXY_PATH,
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(payload),
        },
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on("data", (chunk) => chunks.push(chunk));
        res.on("end", () => {
          resolve({
            statusCode: res.statusCode ?? 0,
            body: Buffer.concat(chunks).toString("utf8"),
          });
        });
      },
    );
    req.on("error", reject);
    req.end(payload);
  });
}
