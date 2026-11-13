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
import {
  CREDENTIAL_KIND_OAUTH,
  createCredentialStore,
} from "../../scripts/mars-credentials.mjs";
import { revokeToken } from "../../scripts/mars-oauth.mjs";

vi.mock("../../scripts/mars-oauth.mjs", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../scripts/mars-oauth.mjs")>()),
  revokeToken: vi.fn(async () => {}),
}));

const VALID_PAT = `dop_v1_${"a".repeat(64)}`;
const OTHER_PAT = `dop_v1_${"b".repeat(64)}`;
const THIRD_PAT = `dop_v1_${"c".repeat(64)}`;

type AuthState = {
  connections: { id: string }[];
  active: { id: string } | null;
};
const API_URL = "https://api.example.test";

/** What harness-api answers for a session with no public URL. */
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
    detachOwnedBy: vi.fn(async () => {}),
    get: vi.fn(() => undefined),
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
      async (
        id: string,
      ): Promise<{ id: string; manifest: unknown } | null> => ({
        id,
        manifest: null,
      }),
    ),
    listConfigSessions: vi.fn(async () => ({
      sessions: [],
      nextPageToken: null,
    })),
    listSandboxSizes: vi.fn(async () => [
      { slug: "mars-2vcpu-4gb", vcpus: 2, memory_mb: 4096 },
    ]),
    createAgentConfig: vi.fn(async (name: string) => ({
      id: "cfg_new",
      name,
    })),
    createSessionFromConfig: vi.fn(async () => ({
      session_id: "sess_new",
      status: "SESSION_STATUS_PROVISIONING",
    })),
    pauseSession: vi.fn(async () => {}),
    resumeSession: vi.fn(async () => {}),
    destroySession: vi.fn(async () => {}),
    deleteAgentConfig: vi.fn(async () => {}),
    getSession: vi.fn(async () => null),
    getAccount: vi.fn(async () => null),
  };
}

/**
 * `realRegistry` keeps the bridge's own tunnel registry, whose listeners bind
 * loopback without dialing anything until a client connects.
 */
function setup({ realRegistry = false } = {}) {
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
    registry: realRegistry ? undefined : registry,
    api,
    credentials,
    ensureAwake,
    env: { MARS_API_BASE_URL: API_URL },
  });
  const ipcMain = fakeIpcMain();
  bridge.registerIpc(ipcMain);
  return {
    registry,
    api,
    credentials,
    ensureAwake,
    bridge,
    ipcMain,
  };
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

  it("openTunnel opens the port-forward tunnel bound to the active connection on the pinned guest port", async () => {
    const { registry, credentials, ipcMain } = setup();
    await ipcMain.invoke(MARS_TUNNEL_IPC.savePat, { token: VALID_PAT });

    const result = await ipcMain.invoke(MARS_TUNNEL_IPC.openTunnel, {
      sessionId: "sess_a",
      localPort: 51000,
    });

    expect(registry.attach).toHaveBeenCalledWith({
      sessionId: "sess_a",
      remotePort: 8000,
      getAccessToken: expect.any(Function),
      apiUrl: API_URL,
      localPort: 51000,
      owner: credentials.getActive()?.id,
    });
    expect(result).toMatchObject({
      sessionId: "sess_a",
      transport: "tunnel",
      host: "http://127.0.0.1:51000",
      localPort: 51000,
    });
  });

  it("a tunnel keeps its own connection's token after switching, and loses it on sign-out", async () => {
    // Arrange
    const { registry, ipcMain } = setup();
    const first = (await ipcMain.invoke(MARS_TUNNEL_IPC.savePat, {
      token: VALID_PAT,
    })) as AuthState;
    const firstId = first.active!.id;
    await ipcMain.invoke(MARS_TUNNEL_IPC.openTunnel, { sessionId: "sess_a" });
    const { getAccessToken } = registry.attach.mock.calls[0][0] as unknown as {
      getAccessToken: () => string | null;
    };

    // Act
    await ipcMain.invoke(MARS_TUNNEL_IPC.savePat, { token: OTHER_PAT });
    const tokenAfterSwitch = getAccessToken();
    await ipcMain.invoke(MARS_TUNNEL_IPC.signOut, firstId);

    // Assert
    expect(tokenAfterSwitch).toBe(VALID_PAT);
    expect(registry.detachOwnedBy).toHaveBeenCalledWith(firstId);
    expect(getAccessToken()).toBeNull();
  });

  it("signOut revokes an OAuth connection's token even when it is not the active one", async () => {
    // Arrange
    const { credentials, ipcMain } = setup();
    const oauth = credentials.save({
      kind: CREDENTIAL_KIND_OAUTH,
      token: "oauth-token",
      label: "OAuth team",
    });
    await ipcMain.invoke(MARS_TUNNEL_IPC.savePat, { token: VALID_PAT });

    // Act
    await ipcMain.invoke(MARS_TUNNEL_IPC.signOut, oauth!.id);

    // Assert
    expect(revokeToken).toHaveBeenCalledWith("oauth-token");
  });

  it("openTunnel refuses to connect while signed out", async () => {
    const { registry, ipcMain } = setup();

    expect(() =>
      ipcMain.invoke(MARS_TUNNEL_IPC.openTunnel, { sessionId: "sess_a" }),
    ).toThrow(/Sign in to DigitalOcean/);
    expect(registry.attach).not.toHaveBeenCalled();
  });

  it("verifies a token and names its team with that token, even if another tab switches teams meanwhile", async () => {
    // DigitalOcean's answers, per token: each belongs to its own team.
    const teamOf: Record<string, string> = {
      [VALID_PAT]: "team-a",
      [OTHER_PAT]: "team-b",
    };
    const calls: { path: string; token: string }[] = [];
    let holdVerify = false;
    let releaseVerify = () => {};
    const verifyHeld = new Promise<void>((resolve) => {
      releaseVerify = resolve;
    });
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async (url, init) => {
        const { pathname } = new URL(String(url));
        const headers = init?.headers as Record<string, string>;
        const token = headers.Authorization.replace(/^Bearer /, "");
        calls.push({ path: pathname, token });
        if (pathname === "/v2/agents/sessions" && holdVerify) {
          holdVerify = false;
          await verifyHeld;
        }
        const team = teamOf[token];
        const body =
          pathname === "/v2/account"
            ? {
                account: {
                  email: `${team}@example.com`,
                  team: { uuid: team, name: team },
                },
              }
            : { sessions: [] };
        return new Response(JSON.stringify(body), {
          headers: { "Content-Type": "application/json" },
        });
      });
    try {
      const credentials = createCredentialStore({
        userDataPath: mkdtempSync(join(tmpdir(), "mars-bridge-")),
      });
      const bridge = createMarsTunnelBridge({
        registry: fakeRegistry(),
        credentials,
        env: { MARS_API_BASE_URL: API_URL },
      });
      await bridge.savePat({ token: OTHER_PAT });
      const teamB = credentials.getActive()!.id;

      holdVerify = true;
      const savingA = bridge.savePat({ token: VALID_PAT });
      await vi.waitFor(() => expect(calls.at(-1)?.token).toBe(VALID_PAT));
      bridge.setActiveConnection(teamB);
      releaseVerify();
      await savingA;

      expect(
        calls.filter((c) => c.path === "/v2/account").map((c) => c.token),
      ).toEqual([OTHER_PAT, VALID_PAT]);
      expect(credentials.getToken(teamB)).toBe(OTHER_PAT);
      expect(credentials.getActive()?.id).toBe(teamB);
      const teamA = credentials
        .list()
        .find(
          (c: { id: string; teamName: string | null }) =>
            c.teamName === "team-a",
        );
      expect(teamA && credentials.getToken(teamA.id)).toBe(VALID_PAT);
    } finally {
      fetchSpy.mockRestore();
    }
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

  it("a rejected token leaves the previously active connection active", async () => {
    const { api, ipcMain } = setup();
    await ipcMain.invoke(MARS_TUNNEL_IPC.savePat, { token: VALID_PAT });
    const second = (await ipcMain.invoke(MARS_TUNNEL_IPC.savePat, {
      token: OTHER_PAT,
    })) as AuthState;
    api.verifyAccess.mockRejectedValueOnce(new Error("403 mars_preview"));

    await expect(
      ipcMain.invoke(MARS_TUNNEL_IPC.savePat, { token: THIRD_PAT }),
    ).rejects.toThrow(/mars_preview/);
    const state = (await ipcMain.invoke(
      MARS_TUNNEL_IPC.getAuthState,
    )) as AuthState;

    expect(state.connections).toHaveLength(2);
    expect(state.active?.id).toBe(second.active?.id);
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
      expect.objectContaining({ api, sessionId: "sess_new" }),
    );
    expect(session).toMatchObject({ status: "SESSION_STATUS_READY" });
  });

  it("openTunnel right after createSession does not poll the session again, but does once it was paused", async () => {
    const { ensureAwake, ipcMain, bridge } = setup({ realRegistry: true });
    await ipcMain.invoke(MARS_TUNNEL_IPC.savePat, { token: VALID_PAT });

    try {
      await ipcMain.invoke(MARS_TUNNEL_IPC.createSession, "cfg_1", "agent-1");
      const tunnel = await ipcMain.invoke(MARS_TUNNEL_IPC.openTunnel, {
        sessionId: "sess_new",
      });
      expect(tunnel).toMatchObject({ status: "connected" });
      expect(ensureAwake).toHaveBeenCalledTimes(1);

      await ipcMain.invoke(MARS_TUNNEL_IPC.pauseSession, "sess_new");
      await ipcMain.invoke(MARS_TUNNEL_IPC.openTunnel, {
        sessionId: "sess_new",
      });
      expect(ensureAwake).toHaveBeenCalledTimes(2);
    } finally {
      await bridge.dispose();
    }
  });

  it("reusing a live tunnel right after createSession still checks the session", async () => {
    // Only the first open after createSession skips the poll: a session
    // paused outside this app (idle, another client) must still be woken.
    const { ensureAwake, ipcMain, bridge } = setup({ realRegistry: true });
    await ipcMain.invoke(MARS_TUNNEL_IPC.savePat, { token: VALID_PAT });

    try {
      await ipcMain.invoke(MARS_TUNNEL_IPC.createSession, "cfg_1", "agent-1");
      await ipcMain.invoke(MARS_TUNNEL_IPC.openTunnel, {
        sessionId: "sess_new",
      });
      await ipcMain.invoke(MARS_TUNNEL_IPC.openTunnel, {
        sessionId: "sess_new",
      });
      expect(ensureAwake).toHaveBeenCalledTimes(2);
    } finally {
      await bridge.dispose();
    }
  });

  it("openTunnel after closeTunnel polls the session again", async () => {
    const { ensureAwake, ipcMain, bridge } = setup({ realRegistry: true });
    await ipcMain.invoke(MARS_TUNNEL_IPC.savePat, { token: VALID_PAT });

    try {
      await ipcMain.invoke(MARS_TUNNEL_IPC.createSession, "cfg_1", "agent-1");
      await ipcMain.invoke(MARS_TUNNEL_IPC.closeTunnel, "sess_new");
      await ipcMain.invoke(MARS_TUNNEL_IPC.openTunnel, {
        sessionId: "sess_new",
      });
      expect(ensureAwake).toHaveBeenCalledTimes(2);
    } finally {
      await bridge.dispose();
    }
  });

  it("createOpenHandsAgent creates a config from the OpenHands manifest and lists it without re-reading", async () => {
    // Arrange
    const { api, ipcMain } = setup();
    await ipcMain.invoke(MARS_TUNNEL_IPC.savePat, { token: VALID_PAT });
    api.listAgentConfigs.mockResolvedValue({
      configs: [{ id: "cfg_new", name: "my-agent" }],
      nextPageToken: null,
    });

    // Act — a renderer passing an LLM key gets no secret into the manifest.
    const created = await ipcMain.invoke(MARS_TUNNEL_IPC.createOpenHandsAgent, {
      name: " my-agent ",
      llmApiKey: "sk-test",
    });
    const page = await ipcMain.invoke(MARS_TUNNEL_IPC.listAgentConfigs);

    // Assert
    expect(api.createAgentConfig).toHaveBeenCalledWith(
      "my-agent",
      ["agent: openhands", "template: openhands", "keep_warm: false", ""].join(
        "\n",
      ),
    );
    expect(created).toMatchObject({ id: "cfg_new", agent: "openhands" });
    expect(page).toMatchObject({
      configs: [{ id: "cfg_new", agent: "openhands" }],
    });
    expect(api.getAgentConfig).not.toHaveBeenCalled();
  });

  it("createOpenHandsAgent writes the chosen sandbox size into the manifest", async () => {
    const { api, ipcMain } = setup();
    await ipcMain.invoke(MARS_TUNNEL_IPC.savePat, { token: VALID_PAT });

    await ipcMain.invoke(MARS_TUNNEL_IPC.createOpenHandsAgent, {
      name: "my-agent",
      size: "mars-16vcpu-32gb",
    });

    expect(api.createAgentConfig).toHaveBeenCalledWith(
      "my-agent",
      expect.stringMatching(/^size: mars-16vcpu-32gb$/m),
    );
  });

  it("createOpenHandsAgent refuses a size that is not a slug", async () => {
    const { api, ipcMain } = setup();
    await ipcMain.invoke(MARS_TUNNEL_IPC.savePat, { token: VALID_PAT });

    await expect(
      ipcMain.invoke(MARS_TUNNEL_IPC.createOpenHandsAgent, {
        name: "my-agent",
        size: "x\nkeep_warm: true",
      }),
    ).rejects.toMatchObject({ status: 400 });
    expect(api.createAgentConfig).not.toHaveBeenCalled();
  });

  it("listSandboxSizes needs a signed-in connection", async () => {
    const { api, ipcMain } = setup();

    await expect(
      ipcMain.invoke(MARS_TUNNEL_IPC.listSandboxSizes),
    ).rejects.toThrow(/Sign in to DigitalOcean/);
    expect(api.listSandboxSizes).not.toHaveBeenCalled();

    await ipcMain.invoke(MARS_TUNNEL_IPC.savePat, { token: VALID_PAT });
    expect(await ipcMain.invoke(MARS_TUNNEL_IPC.listSandboxSizes)).toEqual([
      { slug: "mars-2vcpu-4gb", vcpus: 2, memory_mb: 4096 },
    ]);
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
          ? { agent: "openhands", template: "openhands" }
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

  it("dispose() tears down every tunnel", async () => {
    const { registry, bridge, ipcMain } = setup();
    await ipcMain.invoke(MARS_TUNNEL_IPC.savePat, { token: VALID_PAT });
    await ipcMain.invoke(MARS_TUNNEL_IPC.openTunnel, { sessionId: "sess_a" });

    await bridge.dispose();

    expect(registry.detachAll).toHaveBeenCalledTimes(1);
  });

  it("destroySession asks harness-api first and drops the live connection only once it agreed", async () => {
    const { registry, api, ipcMain } = setup();
    await ipcMain.invoke(MARS_TUNNEL_IPC.savePat, { token: VALID_PAT });
    await ipcMain.invoke(MARS_TUNNEL_IPC.openTunnel, { sessionId: "sess_1" });

    await ipcMain.invoke(MARS_TUNNEL_IPC.destroySession, "sess_1");

    expect(registry.detach).toHaveBeenCalledWith("sess_1");
    expect(api.destroySession).toHaveBeenCalledWith("sess_1");
    expect(api.destroySession.mock.invocationCallOrder[0]).toBeLessThan(
      registry.detach.mock.invocationCallOrder[0],
    );
  });

  it("deleteAgentConfig deletes the config and forgets its cached agent", async () => {
    const { api, ipcMain } = setup();
    await ipcMain.invoke(MARS_TUNNEL_IPC.savePat, { token: VALID_PAT });
    api.listAgentConfigs.mockResolvedValue({
      configs: [{ id: "cfg_1" }],
      nextPageToken: null,
    });
    await ipcMain.invoke(MARS_TUNNEL_IPC.listAgentConfigs);

    await ipcMain.invoke(MARS_TUNNEL_IPC.deleteAgentConfig, "cfg_1");
    await ipcMain.invoke(MARS_TUNNEL_IPC.listAgentConfigs);

    expect(api.deleteAgentConfig).toHaveBeenCalledWith("cfg_1");
    expect(api.getAgentConfig).toHaveBeenCalledTimes(2);
  });

  it("destroySession keeps the connection when harness-api refuses (409/423)", async () => {
    const { registry, api, ipcMain } = setup();
    api.destroySession.mockRejectedValueOnce(
      Object.assign(new Error("a checkpoint is in progress"), { status: 409 }),
    );
    await ipcMain.invoke(MARS_TUNNEL_IPC.savePat, { token: VALID_PAT });
    await ipcMain.invoke(MARS_TUNNEL_IPC.openTunnel, { sessionId: "sess_1" });

    await expect(
      ipcMain.invoke(MARS_TUNNEL_IPC.destroySession, "sess_1"),
    ).rejects.toMatchObject({ status: 409 });

    expect(registry.detach).not.toHaveBeenCalled();
  });
});
