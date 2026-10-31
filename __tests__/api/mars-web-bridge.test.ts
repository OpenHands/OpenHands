import { afterEach, describe, expect, it, vi } from "vitest";

import {
  MarsWebBridgeError,
  createMarsWebBridge,
  getMarsBridgeHeaders,
  installMarsWebBridge,
  probeMarsWebBridge,
} from "#/api/mars/mars-web-bridge";
import type { Backend } from "#/api/backend-registry/types";

const bakedSessionApiKey = vi.hoisted(() => ({ value: null as string | null }));
/** `VITE_BACKEND_BASE_URL`; unset means the page's own origin. */
const configuredBackendUrl = vi.hoisted(() => ({
  value: null as string | null,
}));
const registeredBackends = vi.hoisted(() => ({ value: [] as Backend[] }));
const activeLocalBackend = vi.hoisted(() => ({
  value: null as Backend | null,
}));

// Only the baked key and the configured backend are faked; the origin match
// is the real one, against the jsdom page origin.
vi.mock("#/api/agent-server-config", async (importOriginal) => ({
  ...(await importOriginal<typeof import("#/api/agent-server-config")>()),
  getBakedSessionApiKey: () => bakedSessionApiKey.value,
  getAgentServerBaseUrl: () =>
    configuredBackendUrl.value ?? window.location.origin,
}));

vi.mock("#/api/backend-registry/active-store", () => ({
  getEffectiveLocalBackend: () => activeLocalBackend.value,
  getRegisteredBackends: () => registeredBackends.value,
}));

const PAGE_ORIGIN = window.location.origin;

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
  delete window.marsBridge;
  bakedSessionApiKey.value = null;
  configuredBackendUrl.value = null;
  registeredBackends.value = [];
  activeLocalBackend.value = null;
});

describe("createMarsWebBridge", () => {
  it("turns each bridge call into one POST /mars/rpc/<method> carrying the arguments", async () => {
    const fetchImpl = vi.fn(async () =>
      jsonResponse(200, { result: { sessions: [], nextPageToken: null } }),
    );
    const bridge = createMarsWebBridge(
      "",
      fetchImpl as unknown as typeof fetch,
    );

    const page = await bridge.listConfigSessions("cfg_1", { pageSize: 5 });

    expect(fetchImpl).toHaveBeenCalledWith("/mars/rpc/listConfigSessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ args: ["cfg_1", { pageSize: 5 }] }),
    });
    expect(page).toEqual({ sessions: [], nextPageToken: null });
  });

  it("reads the sandbox size catalog over RPC", async () => {
    const sizes = [{ slug: "mars-2vcpu-4gb", vcpus: 2, memory_mb: 4096 }];
    const fetchImpl = vi.fn(async () => jsonResponse(200, { result: sizes }));
    const bridge = createMarsWebBridge(
      "",
      fetchImpl as unknown as typeof fetch,
    );

    expect(await bridge.listSandboxSizes()).toEqual(sizes);
    expect(fetchImpl).toHaveBeenCalledWith(
      "/mars/rpc/listSandboxSizes",
      expect.objectContaining({ body: JSON.stringify({ args: [] }) }),
    );
  });

  it("raises the server's error message and status", async () => {
    const fetchImpl = vi.fn(async () =>
      jsonResponse(401, {
        error: { message: "Sign in to DigitalOcean first.", status: 401 },
      }),
    );
    const bridge = createMarsWebBridge(
      "",
      fetchImpl as unknown as typeof fetch,
    );

    await expect(bridge.listSessions()).rejects.toMatchObject({
      name: "MarsWebBridgeError",
      message: "Sign in to DigitalOcean first.",
      status: 401,
    });
  });

  it("sends the page's session key on every call so the server can admit it", async () => {
    bakedSessionApiKey.value = "sk-page";
    const fetchImpl = vi.fn(async () => jsonResponse(200, { result: null }));
    const bridge = createMarsWebBridge(
      "",
      fetchImpl as unknown as typeof fetch,
    );

    await bridge.pauseSession("s1");

    expect(fetchImpl).toHaveBeenCalledWith("/mars/rpc/pauseSession", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Session-API-Key": "sk-page",
      },
      body: JSON.stringify({ args: ["s1"] }),
    });
  });

  it("sends the typed session key when no baked key exists on the page", async () => {
    bakedSessionApiKey.value = null;
    registeredBackends.value = [
      {
        id: "default-local",
        name: "Local",
        host: PAGE_ORIGIN,
        apiKey: "qa-typed-session-key",
        kind: "local",
      },
    ];

    expect(getMarsBridgeHeaders()).toEqual({
      "X-Session-API-Key": "qa-typed-session-key",
    });

    const fetchImpl = vi.fn(async () => jsonResponse(200, { result: null }));
    const bridge = createMarsWebBridge(
      "",
      fetchImpl as unknown as typeof fetch,
    );

    await bridge.pauseSession("s1");

    expect(fetchImpl).toHaveBeenCalledWith("/mars/rpc/pauseSession", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Session-API-Key": "qa-typed-session-key",
      },
      body: JSON.stringify({ args: ["s1"] }),
    });
  });

  it("prefers the active backend's key over an older one saved for the same origin", () => {
    const current: Backend = {
      id: "current",
      name: "Current",
      host: PAGE_ORIGIN,
      apiKey: "current-key",
      kind: "local",
    };
    registeredBackends.value = [
      {
        id: "stale",
        name: "Stale",
        host: PAGE_ORIGIN,
        apiKey: "stale-key",
        kind: "local",
      },
      current,
    ];
    activeLocalBackend.value = current;

    expect(getMarsBridgeHeaders()).toEqual({
      "X-Session-API-Key": "current-key",
    });
  });

  it("never sends a key stored for another host, a DigitalOcean session or a cloud backend", async () => {
    const marsSession: Backend = {
      id: "mars-1",
      name: "MARS Session",
      host: `${PAGE_ORIGIN}/mars/sessions/s1`,
      apiKey: "session-token",
      kind: "local",
      marsSessionId: "s1",
    };
    registeredBackends.value = [
      {
        id: "cloud-1",
        name: "Cloud",
        host: PAGE_ORIGIN,
        apiKey: "cloud-key",
        kind: "cloud",
      },
      {
        id: "remote-1",
        name: "Remote",
        host: "https://remote.openhands.dev",
        apiKey: "remote-key",
        kind: "local",
      },
      marsSession,
    ];
    activeLocalBackend.value = marsSession;
    const fetchImpl = vi.fn(async () => jsonResponse(200, { result: null }));
    const bridge = createMarsWebBridge(
      "",
      fetchImpl as unknown as typeof fetch,
    );

    await bridge.pauseSession("s1");

    expect(fetchImpl).toHaveBeenCalledWith("/mars/rpc/pauseSession", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ args: ["s1"] }),
    });
  });

  it("never sends the page's key to a bridge on a foreign host", async () => {
    bakedSessionApiKey.value = "sk-page";
    const fetchImpl = vi.fn(async () => jsonResponse(200, { result: null }));
    const foreignBridge = createMarsWebBridge(
      "https://foreign-host.com",
      fetchImpl as unknown as typeof fetch,
    );

    await foreignBridge.pauseSession("s2");

    expect(fetchImpl).toHaveBeenCalledWith(
      "https://foreign-host.com/mars/rpc/pauseSession",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ args: ["s2"] }),
      },
    );
  });

  it("keeps the key of a backend the build points elsewhere off the page's own bridge", () => {
    configuredBackendUrl.value = "https://remote.openhands.dev";
    bakedSessionApiKey.value = "remote-baked-key";
    registeredBackends.value = [
      {
        id: "default-local",
        name: "Local",
        host: "https://remote.openhands.dev",
        apiKey: "remote-key",
        kind: "local",
      },
    ];

    expect(getMarsBridgeHeaders()).toEqual({});

    registeredBackends.value = [
      ...registeredBackends.value,
      {
        id: "canvas",
        name: "Canvas",
        host: `${PAGE_ORIGIN}/`,
        apiKey: "canvas-key",
        kind: "local",
      },
    ];
    expect(getMarsBridgeHeaders()).toEqual({
      "X-Session-API-Key": "canvas-key",
    });
  });

  it("never offers OAuth", async () => {
    const bridge = createMarsWebBridge("", vi.fn() as unknown as typeof fetch);
    await expect(bridge.signInWithOAuth()).rejects.toBeInstanceOf(
      MarsWebBridgeError,
    );
  });
});

describe("probeMarsWebBridge / installMarsWebBridge", () => {
  it("reports whether the server hosts the bridge", async () => {
    const ok = vi.fn(async () => jsonResponse(200, { ok: true }));
    const missing = vi.fn(async () => new Response("", { status: 404 }));
    // A server that answers every path with index.html is a 200 too.
    const spa = vi.fn(
      async () =>
        new Response("<!doctype html><title>Canvas</title>", {
          status: 200,
          headers: { "Content-Type": "text/html" },
        }),
    );
    const down = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    });

    expect(await probeMarsWebBridge("", ok as unknown as typeof fetch)).toBe(
      true,
    );
    expect(
      await probeMarsWebBridge("", missing as unknown as typeof fetch),
    ).toBe(false);
    expect(await probeMarsWebBridge("", spa as unknown as typeof fetch)).toBe(
      false,
    );
    expect(await probeMarsWebBridge("", down as unknown as typeof fetch)).toBe(
      false,
    );
  });

  it("installs window.marsBridge only when the server answers, and leaves an existing bridge alone", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("", { status: 404 })),
    );
    expect(await installMarsWebBridge()).toBe(false);
    expect(window.marsBridge).toBeUndefined();

    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonResponse(200, { ok: true })),
    );
    expect(await installMarsWebBridge()).toBe(true);
    expect(window.marsBridge).toBeDefined();

    const electronBridge = window.marsBridge;
    expect(await installMarsWebBridge()).toBe(false);
    expect(window.marsBridge).toBe(electronBridge);
  });
});
