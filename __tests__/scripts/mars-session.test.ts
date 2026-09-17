// @vitest-environment node
import { createServer } from "node:http";
import net from "node:net";
import { afterEach, describe, expect, it } from "vitest";

import { ensureSessionAwake } from "../../scripts/mars-session.mjs";

const sessionId = "sess_abc123";
const accessToken = "test-token";

/**
 * A fake harness-api sessions endpoint. `statusSequence` is popped from the
 * front on each GET (repeating the last entry once exhausted), so a test can
 * script e.g. ["SESSION_STATUS_PAUSED", "SESSION_STATUS_READY"] to simulate
 * the session coming up after a resume call.
 */
function startFakeSessionsApi(statusSequence: string[], { expectToken = accessToken } = {}) {
  let resumeCalls = 0;
  let getCalls = 0;

  const server = createServer((req, res) => {
    if (req.headers.authorization !== `Bearer ${expectToken}`) {
      res.writeHead(401).end();
      return;
    }
    if (req.method === "GET" && req.url === `/v2/agents/sessions/${sessionId}`) {
      getCalls += 1;
      const status = statusSequence[Math.min(getCalls - 1, statusSequence.length - 1)];
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ session: { session_id: sessionId, status } }));
      return;
    }
    if (req.method === "POST" && req.url === `/v2/agents/sessions/${sessionId}/resume`) {
      resumeCalls += 1;
      res.writeHead(204).end();
      return;
    }
    res.writeHead(404).end();
  });

  return new Promise<{ port: number; close: () => void; resumeCalls: () => number; getCalls: () => number }>(
    (resolve) => {
      server.listen(0, "127.0.0.1", () => {
        const { port } = server.address() as net.AddressInfo;
        resolve({
          port,
          close: () => server.close(),
          resumeCalls: () => resumeCalls,
          getCalls: () => getCalls,
        });
      });
    },
  );
}

describe("ensureSessionAwake", () => {
  let api: Awaited<ReturnType<typeof startFakeSessionsApi>> | undefined;

  afterEach(() => {
    api?.close();
    api = undefined;
  });

  it("resolves immediately for an already-ready session, without resuming", async () => {
    api = await startFakeSessionsApi(["SESSION_STATUS_READY"]);

    const session = await ensureSessionAwake({
      apiUrl: `http://127.0.0.1:${api.port}`,
      sessionId,
      accessToken,
    });

    expect(session.status).toBe("SESSION_STATUS_READY");
    expect(api.resumeCalls()).toBe(0);
  });

  it("resumes a paused session and waits for it to become ready", async () => {
    api = await startFakeSessionsApi([
      "SESSION_STATUS_PAUSED",
      "SESSION_STATUS_PAUSED",
      "SESSION_STATUS_READY",
    ]);

    const session = await ensureSessionAwake({
      apiUrl: `http://127.0.0.1:${api.port}`,
      sessionId,
      accessToken,
      pollIntervalMs: 10,
    });

    expect(session.status).toBe("SESSION_STATUS_READY");
    expect(api.resumeCalls()).toBe(1);
    expect(api.getCalls()).toBeGreaterThanOrEqual(3);
  });

  it("rejects immediately for a terminal session", async () => {
    api = await startFakeSessionsApi(["SESSION_STATUS_FAILED"]);

    await expect(
      ensureSessionAwake({
        apiUrl: `http://127.0.0.1:${api.port}`,
        sessionId,
        accessToken,
        pollIntervalMs: 10,
      }),
    ).rejects.toThrow(/SESSION_STATUS_FAILED/);
    expect(api.resumeCalls()).toBe(0);
  });

  it("rejects if the session becomes terminal while resuming", async () => {
    api = await startFakeSessionsApi(["SESSION_STATUS_PAUSED", "SESSION_STATUS_DESTROYED"]);

    await expect(
      ensureSessionAwake({
        apiUrl: `http://127.0.0.1:${api.port}`,
        sessionId,
        accessToken,
        pollIntervalMs: 10,
      }),
    ).rejects.toThrow(/became SESSION_STATUS_DESTROYED/);
  });

  it("times out if the session never becomes ready", async () => {
    api = await startFakeSessionsApi(["SESSION_STATUS_PAUSED"]);

    await expect(
      ensureSessionAwake({
        apiUrl: `http://127.0.0.1:${api.port}`,
        sessionId,
        accessToken,
        pollIntervalMs: 10,
        timeoutMs: 50,
      }),
    ).rejects.toThrow(/Timed out waiting/);
  });

  it("propagates an HTTP error from the get-session call", async () => {
    api = await startFakeSessionsApi(["SESSION_STATUS_READY"], { expectToken: "other-token" });

    await expect(
      ensureSessionAwake({
        apiUrl: `http://127.0.0.1:${api.port}`,
        sessionId,
        accessToken,
      }),
    ).rejects.toThrow(/Failed to get session.*HTTP 401/);
  });
});
