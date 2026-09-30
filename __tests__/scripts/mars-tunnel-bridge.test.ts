// @vitest-environment node
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";

import {
  MARS_TUNNEL_IPC,
  createMarsTunnelBridge,
  readMarsConfig,
  readMarsDefaults,
} from "../../scripts/mars-tunnel-bridge.mjs";
import { createCredentialStore } from "../../scripts/mars-credentials.mjs";

const VALID_PAT = `dop_v1_${"a".repeat(64)}`;
const API_URL = "https://api.example.test";

/** A fake ipcMain that just records handlers so tests can invoke them directly. */
function fakeIpcMain() {
  const handlers = new Map<
    string,
    (event: unknown, ...args: unknown[]) => unknown
  >();
  return {
    channels: () => [...handlers.keys()],
    handle: vi.fn(
      (
        channel: string,
        fn: (event: unknown, ...args: unknown[]) => unknown,
      ) => {
        handlers.set(channel, fn);
      },
    ),
    invoke: (channel: string, ...args: unknown[]) => {
      const handler = handlers.get(channel);
      if (!handler) throw new Error(`no handler registered for ${channel}`);
      return handler({}, ...args);
    },
  };
}

function fakeRegistry() {
  return {
    attach: vi.fn(
      async (params: { sessionId: string; remotePort: number }) => ({
        ...params,
        localPort: 51000,
        status: "connected" as const,
        error: undefined,
        upstreamFailure: null,
      }),
    ),
    detach: vi.fn(async () => {}),
    detachAll: vi.fn(async () => {}),
    get: vi.fn(() => undefined),
    list: vi.fn(() => []),
  };
}

function fakeApi() {
  return {
    verifyAccess: vi.fn(async () => {}),
    listSessions: vi.fn(async () => ({ sessions: [], nextPageToken: null })),
    listAgentConfigs: vi.fn(async () => ({
      configs: [] as { id: string; name?: string }[],
      nextPageToken: null,
    })),
    getAgentConfig: vi.fn(
      async (id: string): Promise<{ id: string; manifest: unknown } | null> => ({
        id,
        manifest: null,
      }),
    ),
    listConfigSessions: vi.fn(async () => ({
      sessions: [],
      nextPageToken: null,
    })),
    createSessionFromConfig: vi.fn(async () => ({
      session_id: "sess_new",
      status: "SESSION_STATUS_PROVISIONING",
    })),
    pauseSession: vi.fn(async () => {}),
    resumeSession: vi.fn(async () => {}),
    getSession: vi.fn(async () => null),
  };
}

function setup() {
  const registry = fakeRegistry();
  const api = fakeApi();
  const credentials = createCredentialStore({
    userDataPath: mkdtempSync(join(tmpdir(), "mars-bridge-")),
  });
  const ensureAwake = vi.fn(async ({ sessionId }: { sessionId: string }) => ({
    session_id: sessionId,
    status: "SESSION_STATUS_READY",
  }));
  const bridge = createMarsTunnelBridge({
    registry,
    api,
    credentials,
    ensureAwake,
    env: { MARS_API_BASE_URL: API_URL },
  });
  const ipcMain = fakeIpcMain();
  bridge.registerIpc(ipcMain);
  return { registry, api, credentials, ensureAwake, bridge, ipcMain };
}

describe("readMarsConfig", () => {
  it.each([
    {
      env: { DO_OAUTH_CLIENT_ID: "env-client" },
      defaults: { oauthClientId: "shipped-client" },
      expected: "env-client",
    },
    {
      env: {},
      defaults: { oauthClientId: "shipped-client" },
      expected: "shipped-client",
    },
    { env: {}, defaults: { oauthClientId: "" }, expected: null },
  ])(
    "resolves the OAuth client id with env over shipped defaults ($expected)",
    ({ env, defaults, expected }) => {
      expect(readMarsConfig(env, defaults).oauthClientId).toBe(expected);
    },
  );

  it("reads the shipped defaults from config/defaults.json", () => {
    expect(readMarsDefaults()).toHaveProperty("apiBaseUrl");
  });
});

describe("createMarsTunnelBridge", () => {
  it("registers a handler for every MARS channel", () => {
    const { ipcMain } = setup();

    expect(ipcMain.channels().sort()).toEqual(
      Object.values(MARS_TUNNEL_IPC).sort(),
    );
  });

  it("openTunnel sources the token from the active connection and pins the guest port", async () => {
    const { registry, ipcMain } = setup();
    await ipcMain.invoke(MARS_TUNNEL_IPC.savePat, { token: VALID_PAT });

    const result = await ipcMain.invoke(MARS_TUNNEL_IPC.openTunnel, {
      sessionId: "sess_a",
      localPort: 51000,
    });

    expect(registry.attach).toHaveBeenCalledWith({
      sessionId: "sess_a",
      remotePort: 8000,
      accessToken: VALID_PAT,
      apiUrl: API_URL,
      localPort: 51000,
    });
    expect(result).toMatchObject({ sessionId: "sess_a", localPort: 51000 });
  });

  it("openTunnel refuses to dial while signed out", async () => {
    const { registry, ipcMain } = setup();

    expect(() =>
      ipcMain.invoke(MARS_TUNNEL_IPC.openTunnel, { sessionId: "sess_a" }),
    ).toThrow(/Sign in to DigitalOcean/);
    expect(registry.attach).not.toHaveBeenCalled();
  });

  it("savePat keeps a token only once the API accepts it", async () => {
    const { api, ipcMain } = setup();
    api.verifyAccess.mockRejectedValueOnce(new Error("403 mars_preview"));

    await expect(
      ipcMain.invoke(MARS_TUNNEL_IPC.savePat, { token: VALID_PAT }),
    ).rejects.toThrow(/mars_preview/);
    const afterRejection = await ipcMain.invoke(MARS_TUNNEL_IPC.getAuthState);
    const afterSuccess = await ipcMain.invoke(MARS_TUNNEL_IPC.savePat, {
      token: VALID_PAT,
    });

    expect(afterRejection).toMatchObject({ connections: [], active: null });
    expect(afterSuccess).toMatchObject({
      connections: [expect.objectContaining({ kind: "pat" })],
      active: expect.objectContaining({ kind: "pat" }),
    });
  });

  it("savePat rejects a malformed token without calling the API", async () => {
    const { api, ipcMain } = setup();

    await expect(
      ipcMain.invoke(MARS_TUNNEL_IPC.savePat, { token: "not-a-token" }),
    ).rejects.toThrow(/personal access token/);
    expect(api.verifyAccess).not.toHaveBeenCalled();
  });

  it("createSession resolves only once the new session is ready", async () => {
    const { api, ensureAwake, ipcMain } = setup();
    await ipcMain.invoke(MARS_TUNNEL_IPC.savePat, { token: VALID_PAT });

    const session = await ipcMain.invoke(
      MARS_TUNNEL_IPC.createSession,
      "cfg_1",
      "agent-abc123",
    );

    expect(api.createSessionFromConfig).toHaveBeenCalledWith(
      "cfg_1",
      "agent-abc123",
    );
    expect(ensureAwake).toHaveBeenCalledWith(
      expect.objectContaining({
        sessionId: "sess_new",
        accessToken: VALID_PAT,
        apiUrl: API_URL,
      }),
    );
    expect(session).toMatchObject({ status: "SESSION_STATUS_READY" });
  });

  it("listAgentConfigs names each config's agent from its manifest, reading each once", async () => {
    const { api, ipcMain } = setup();
    api.listAgentConfigs.mockResolvedValue({
      configs: [{ id: "cfg_flat" }, { id: "cfg_spec" }],
      nextPageToken: null,
    });
    api.getAgentConfig.mockImplementation(async (id: string) => ({
      id,
      manifest:
        id === "cfg_flat"
          ? { agent: "openhands", template: "openhands-poc" }
          : { kind: "Agent", spec: { agent: "Claude-Code" } },
    }));

    await ipcMain.invoke(MARS_TUNNEL_IPC.listAgentConfigs);
    const page = await ipcMain.invoke(MARS_TUNNEL_IPC.listAgentConfigs);

    expect(page).toMatchObject({
      configs: [
        { id: "cfg_flat", agent: "openhands" },
        { id: "cfg_spec", agent: "claude-code" },
      ],
    });
    expect(api.getAgentConfig).toHaveBeenCalledTimes(2);
  });

  it("dispose() tears down every tunnel via registry.detachAll", async () => {
    const { registry, bridge } = setup();

    await bridge.dispose();

    expect(registry.detachAll).toHaveBeenCalledTimes(1);
  });
});
