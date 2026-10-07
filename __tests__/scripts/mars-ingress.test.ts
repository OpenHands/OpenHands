// @vitest-environment node
import { createServer } from "node:http";
import net from "node:net";
import { afterEach, describe, expect, it } from "vitest";

import { createMarsApiClient } from "../../scripts/mars-api.mjs";
import {
  MarsIngressUnsupportedError,
  resolveIngressURL,
} from "../../scripts/mars-ingress.mjs";

const sessionId = "sess_ing";
const accessToken = "test-token";
const INGRESS_URL = "https://ing-1.nyc3.sandbox.ondigitalocean.com";

/**
 * A fake harness-api with the session + ingress endpoints. The session starts
 * in `sessionStatus` and flips to READY once resumed. `ingressStates` is
 * consumed one per ingress GET (repeating the last), so a test can script
 * PENDING → READY; `ingressStatus` replaces the ingress answer with an error
 * status (501, 401, …). The ingress endpoint answers 409 while the session is
 * not READY, as harness-api does.
 */
function startFakeHarness({
  sessionStatus = "SESSION_STATUS_READY",
  ingressStates = ["INGRESS_URL_STATE_READY"],
  ingressStatus = 200,
} = {}) {
  let status = sessionStatus;
  let resumeCalls = 0;
  let ingressCalls = 0;
  let ingressBeforeReady = 0;

  const server = createServer((req, res) => {
    if (req.headers.authorization !== `Bearer ${accessToken}`) {
      res.writeHead(401).end();
      return;
    }
    if (
      req.method === "GET" &&
      req.url === `/v2/agents/sessions/${sessionId}`
    ) {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ session: { session_id: sessionId, status } }));
      return;
    }
    if (
      req.method === "POST" &&
      req.url === `/v2/agents/sessions/${sessionId}/resume`
    ) {
      resumeCalls += 1;
      status = "SESSION_STATUS_READY";
      res.writeHead(204).end();
      return;
    }
    if (
      req.method === "GET" &&
      req.url === `/v2/agents/sessions/${sessionId}/ingress`
    ) {
      ingressCalls += 1;
      if (status !== "SESSION_STATUS_READY") {
        ingressBeforeReady += 1;
        res.writeHead(409, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({ error: { message: "session is not running" } }),
        );
        return;
      }
      if (ingressStatus !== 200) {
        res.writeHead(ingressStatus, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({ error: { message: `ingress ${ingressStatus}` } }),
        );
        return;
      }
      const state =
        ingressStates[Math.min(ingressCalls - 1, ingressStates.length - 1)];
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          ingress_url: {
            ingress_url_id: "ing-1",
            session_id: sessionId,
            port: 8000,
            url: INGRESS_URL,
            state,
          },
        }),
      );
      return;
    }
    res.writeHead(404).end();
  });

  return new Promise<{
    api: ReturnType<typeof createMarsApiClient>;
    close: () => void;
    resumeCalls: () => number;
    ingressCalls: () => number;
    ingressBeforeReady: () => number;
  }>((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address() as net.AddressInfo;
      resolve({
        api: createMarsApiClient({
          baseUrl: `http://127.0.0.1:${port}`,
          getToken: async () => accessToken,
        }),
        close: () => {
          server.closeAllConnections();
          server.close();
        },
        resumeCalls: () => resumeCalls,
        ingressCalls: () => ingressCalls,
        ingressBeforeReady: () => ingressBeforeReady,
      });
    });
  });
}

describe("resolveIngressURL", () => {
  let harness: Awaited<ReturnType<typeof startFakeHarness>> | undefined;

  afterEach(() => {
    harness?.close();
    harness = undefined;
  });

  it("returns the READY public URL of a running session without resuming it", async () => {
    harness = await startFakeHarness();

    const ingress = await resolveIngressURL({
      api: harness.api,
      sessionId,
      pollIntervalMs: 10,
    });

    expect(ingress).toEqual({
      url: INGRESS_URL,
      ingressUrlId: "ing-1",
      port: 8000,
      state: "INGRESS_URL_STATE_READY",
    });
    expect(harness.resumeCalls()).toBe(0);
  });

  it("polls a PENDING URL until the gateway route is READY", async () => {
    harness = await startFakeHarness({
      ingressStates: [
        "INGRESS_URL_STATE_PENDING",
        "INGRESS_URL_STATE_PENDING",
        "INGRESS_URL_STATE_READY",
      ],
    });

    const ingress = await resolveIngressURL({
      api: harness.api,
      sessionId,
      pollIntervalMs: 10,
    });

    expect(ingress.state).toBe("INGRESS_URL_STATE_READY");
    expect(harness.ingressCalls()).toBe(3);
  });

  it("wakes a paused session before asking for its URL (409 while paused)", async () => {
    harness = await startFakeHarness({
      sessionStatus: "SESSION_STATUS_PAUSED",
    });

    const ingress = await resolveIngressURL({
      api: harness.api,
      sessionId,
      pollIntervalMs: 10,
    });

    expect(ingress.url).toBe(INGRESS_URL);
    expect(harness.resumeCalls()).toBe(1);
    expect(harness.ingressBeforeReady()).toBe(0);
  });

  it("surfaces a 501 as MarsIngressUnsupportedError so the caller can fall back to the tunnel", async () => {
    harness = await startFakeHarness({ ingressStatus: 501 });

    await expect(
      resolveIngressURL({ api: harness.api, sessionId, pollIntervalMs: 10 }),
    ).rejects.toBeInstanceOf(MarsIngressUnsupportedError);
  });

  it("propagates other harness-api errors unchanged", async () => {
    harness = await startFakeHarness({ ingressStatus: 403 });

    await expect(
      resolveIngressURL({ api: harness.api, sessionId, pollIntervalMs: 10 }),
    ).rejects.toMatchObject({ name: "MarsApiError", status: 403 });
  });

  it("times out when the URL never becomes READY", async () => {
    harness = await startFakeHarness({
      ingressStates: ["INGRESS_URL_STATE_PENDING"],
    });

    await expect(
      resolveIngressURL({
        api: harness.api,
        sessionId,
        pollIntervalMs: 10,
        timeoutMs: 60,
      }),
    ).rejects.toThrow(/Timed out waiting for the public URL/);
  });
});
