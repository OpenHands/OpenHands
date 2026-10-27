import { afterEach, describe, expect, it, vi } from "vitest";

import {
  MaritimeWebBridgeError,
  createMaritimeWebBridge,
  installMaritimeWebBridge,
  probeMaritimeWebBridge,
} from "#/api/maritime/maritime-web-bridge";

const bakedSessionApiKey = vi.hoisted(() => ({ value: null as string | null }));

vi.mock("#/api/agent-server-config", async (importOriginal) => ({
  ...(await importOriginal<typeof import("#/api/agent-server-config")>()),
  getBakedSessionApiKey: () => bakedSessionApiKey.value,
}));

vi.mock("#/api/backend-registry/active-store", () => ({
  getEffectiveLocalBackend: () => null,
  getRegisteredBackends: () => [],
}));

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => {
  bakedSessionApiKey.value = null;
  delete window.maritimeBridge;
  vi.unstubAllGlobals();
});

describe("createMaritimeWebBridge", () => {
  it("posts each call with its arguments and the page's session key", async () => {
    bakedSessionApiKey.value = "page-key";
    const fetchImpl = vi.fn(async () =>
      jsonResponse(200, { result: { agentId: "a1" } }),
    );
    const bridge = createMaritimeWebBridge("", fetchImpl as never);

    await expect(bridge.openAgent("a1")).resolves.toEqual({ agentId: "a1" });

    const [url, init] = fetchImpl.mock.calls[0] as unknown as [
      string,
      RequestInit,
    ];
    expect(url).toBe("/maritime/rpc/openAgent");
    expect(JSON.parse(String(init.body))).toEqual({ args: ["a1"] });
    expect(new Headers(init.headers).get("X-Session-API-Key")).toBe("page-key");
  });

  it("rejects with the server's message and status", async () => {
    const fetchImpl = vi.fn(async () =>
      jsonResponse(429, {
        error: { message: "Maritime limit reached", status: 429 },
      }),
    );
    const error = await createMaritimeWebBridge("", fetchImpl as never)
      .listAgents()
      .catch((e: unknown) => e);

    expect(error).toBeInstanceOf(MaritimeWebBridgeError);
    expect(error).toMatchObject({
      message: "Maritime limit reached",
      status: 429,
    });
  });
});

describe("installMaritimeWebBridge", () => {
  it("installs the bridge only when the server says it hosts one", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonResponse(200, { ok: true })),
    );
    await expect(installMaritimeWebBridge()).resolves.toBe(true);
    expect(window.maritimeBridge).toBeDefined();
  });

  it("stays away from a server that answers every path with the app shell", async () => {
    const fetchImpl = vi.fn(
      async () => new Response("<!doctype html>", { status: 200 }),
    );
    await expect(probeMaritimeWebBridge("", fetchImpl as never)).resolves.toBe(
      false,
    );
  });

  it("leaves a bridge Electron already provided alone", async () => {
    const preload = {} as NonNullable<typeof window.maritimeBridge>;
    window.maritimeBridge = preload;
    const fetchImpl = vi.fn();
    vi.stubGlobal("fetch", fetchImpl);

    await expect(installMaritimeWebBridge()).resolves.toBe(false);
    expect(window.maritimeBridge).toBe(preload);
    expect(fetchImpl).not.toHaveBeenCalled();
  });
});
