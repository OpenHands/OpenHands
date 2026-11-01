// @vitest-environment node
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { beforeEach, describe, expect, it, vi } from "vitest";

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
import {
  OAuthError,
  exchangeCode,
  refreshGrant,
  revokeToken,
} from "../../scripts/mars-oauth.mjs";
import {
  createEncryptedFileStore,
  createMemoryStore,
} from "../../scripts/mars-secret-store.mjs";

vi.mock("../../scripts/mars-oauth.mjs", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../scripts/mars-oauth.mjs")>()),
  revokeToken: vi.fn(async () => true),
  exchangeCode: vi.fn(),
  refreshGrant: vi.fn(),
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
function setup({
  realRegistry = false,
  env = {},
}: { realRegistry?: boolean; env?: Record<string, string> } = {}) {
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
    env: { MARS_API_BASE_URL: API_URL, ...env },
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
      getAccessToken: () => Promise<string | null>;
    };

    // Act
    await ipcMain.invoke(MARS_TUNNEL_IPC.savePat, { token: OTHER_PAT });
    const tokenAfterSwitch = await getAccessToken();
    await ipcMain.invoke(MARS_TUNNEL_IPC.signOut, firstId);

    // Assert
    expect(tokenAfterSwitch).toBe(VALID_PAT);
    expect(registry.detachOwnedBy).toHaveBeenCalledWith(firstId);
    expect(await getAccessToken()).toBeNull();
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

describe("signing in with DigitalOcean (code + PKCE)", () => {
  // The OAuth calls are module mocks: start each case with no calls seen.
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const REDIRECT = "http://127.0.0.1:3000/mars/oauth/callback";
  const WITH_CLIENT = { env: { DO_OAUTH_CLIENT_ID: "client-1" } };
  const inMinutes = (minutes: number) =>
    new Date(Date.now() + minutes * 60_000).toISOString();
  const GRANT = {
    accessToken: "oauth-access",
    refreshToken: "refresh-1",
    accessExpiresAt: inMinutes(60),
    account: { uuid: "team-1", name: "My Team", email: "me@example.com" },
  };
  const stateOf = (authorizeUrl: string) =>
    new URL(authorizeUrl).searchParams.get("state");

  it("finishes a browser sign-in from its return address, named after DigitalOcean's reply", async () => {
    const { bridge, api } = setup(WITH_CLIENT);
    vi.mocked(exchangeCode).mockResolvedValueOnce(GRANT);
    const { attemptId, authorizeUrl } = bridge.beginSignIn({
      redirectUri: REDIRECT,
    });
    const waiting = bridge.awaitSignIn(attemptId);
    // What the return route hands over: its request's path and query.
    const returned = `${new URL(REDIRECT).pathname}?code=code-1&state=${stateOf(authorizeUrl)}`;

    await bridge.completeSignIn(returned);

    expect(exchangeCode).toHaveBeenCalledWith({
      clientId: "client-1",
      code: "code-1",
      verifier: expect.any(String),
      redirectUri: REDIRECT,
    });
    await expect(waiting).resolves.toMatchObject({
      status: "done",
      authState: {
        active: { kind: "oauth", teamName: "My Team", email: "me@example.com" },
      },
    });
    expect(api.getAccount).not.toHaveBeenCalled();
    await expect(bridge.completeSignIn(returned)).rejects.toThrow(
      /no longer open/,
    );
  });

  it("takes a pasted return address only for the attempt it belongs to", async () => {
    const { bridge } = setup(WITH_CLIENT);
    vi.mocked(exchangeCode).mockResolvedValue(GRANT);
    const first = bridge.beginSignIn({ redirectUri: REDIRECT });
    const second = bridge.beginSignIn({ redirectUri: REDIRECT });
    const pasted = `${REDIRECT}?code=code-1&state=${stateOf(first.authorizeUrl)}`;

    await expect(
      bridge.completeSignIn(pasted, second.attemptId),
    ).rejects.toThrow(/isn't from this sign-in/);
    await expect(
      bridge.completeSignIn(pasted, first.attemptId),
    ).resolves.toMatchObject({ active: { kind: "oauth" } });
  });

  it("keeps a sign-in open after a pasted address with no code, for the right one", async () => {
    const { bridge } = setup(WITH_CLIENT);
    vi.mocked(exchangeCode).mockResolvedValue(GRANT);
    const { attemptId, authorizeUrl } = bridge.beginSignIn({
      redirectUri: REDIRECT,
    });
    const state = stateOf(authorizeUrl);

    await expect(
      bridge.completeSignIn(`${REDIRECT}?state=${state}`, attemptId),
    ).rejects.toThrow(/no sign-in code/);
    await expect(
      bridge.completeSignIn(
        `${REDIRECT}?code=code-1&state=${state}`,
        attemptId,
      ),
    ).resolves.toMatchObject({ active: { kind: "oauth" } });
  });

  it("ends the grant of a sign-in that fails after the exchange", async () => {
    const { bridge, api } = setup(WITH_CLIENT);
    vi.mocked(exchangeCode).mockResolvedValueOnce(GRANT);
    // A team without Managed Agents: DigitalOcean signs in, harness-api refuses.
    api.verifyAccess.mockRejectedValueOnce(
      Object.assign(new Error("Managed Agents is not enabled"), {
        status: 403,
      }),
    );
    const { authorizeUrl } = bridge.beginSignIn({ redirectUri: REDIRECT });

    await expect(
      bridge.completeSignIn(
        `${REDIRECT}?code=code-1&state=${stateOf(authorizeUrl)}`,
      ),
    ).rejects.toThrow(/not enabled/);

    expect(revokeToken).toHaveBeenCalledWith("oauth-access");
    expect(bridge.getAuthState().connections).toEqual([]);
  });

  it("ends the OAuth grant a new sign-in to the same team replaces, but never a personal access token", async () => {
    const { bridge, ipcMain, credentials } = setup(WITH_CLIENT);
    // A personal access token for team-1, then an OAuth sign-in for it.
    await ipcMain.invoke(MARS_TUNNEL_IPC.savePat, { token: VALID_PAT });
    credentials.setTeam(credentials.getActive()!.id, { uuid: "team-1" });
    const signIn = async (grant: typeof GRANT) => {
      vi.mocked(exchangeCode).mockResolvedValueOnce(grant);
      const { authorizeUrl } = bridge.beginSignIn({ redirectUri: REDIRECT });
      await bridge.completeSignIn(
        `${REDIRECT}?code=code-1&state=${stateOf(authorizeUrl)}`,
      );
    };

    await signIn(GRANT);
    expect(revokeToken).not.toHaveBeenCalled();

    await signIn({ ...GRANT, accessToken: "oauth-access-2" });

    await vi.waitFor(() =>
      expect(revokeToken).toHaveBeenCalledWith("oauth-access"),
    );
    expect(revokeToken).toHaveBeenCalledTimes(1);
    expect(bridge.getAuthState().connections).toHaveLength(1);
    expect(credentials.getActiveToken()).toBe("oauth-access-2");
  });

  it("keeps the refresh token it has when a refresh reply carries none", async () => {
    const { bridge, registry, credentials } = setup(WITH_CLIENT);
    const saved = credentials.save({
      kind: CREDENTIAL_KIND_OAUTH,
      token: "old-access",
      refreshToken: "refresh-1",
      accessExpiresAt: inMinutes(0.5),
      clientId: "client-1",
      label: "DigitalOcean",
    });
    vi.mocked(refreshGrant).mockResolvedValueOnce({
      ...GRANT,
      accessToken: "new-access",
      refreshToken: null,
    });
    await bridge.openTunnel({ sessionId: "sess_a" });
    const { getAccessToken } = registry.attach.mock.calls[0][0] as unknown as {
      getAccessToken: () => Promise<string | null>;
    };

    await expect(getAccessToken()).resolves.toBe("new-access");
    expect(credentials.getGrant(saved.id)).toMatchObject({
      token: "new-access",
      refreshToken: "refresh-1",
    });
  });

  it("reports a declined sign-in to the waiting page and keeps the old connection", async () => {
    const { bridge, ipcMain } = setup(WITH_CLIENT);
    await ipcMain.invoke(MARS_TUNNEL_IPC.savePat, { token: VALID_PAT });
    const { attemptId, authorizeUrl } = bridge.beginSignIn({
      redirectUri: REDIRECT,
    });

    await expect(
      bridge.completeSignIn(
        `${REDIRECT}?error=access_denied&state=${stateOf(authorizeUrl)}`,
      ),
    ).rejects.toThrow(/access_denied/);

    await expect(bridge.awaitSignIn(attemptId)).resolves.toMatchObject({
      status: "failed",
      message: "access_denied",
    });
    expect(bridge.getAuthState().active).toMatchObject({ kind: "pat" });
  });

  it("says sign-in is not set up when this Canvas has no OAuth client", () => {
    const { bridge } = setup();

    expect(() => bridge.beginSignIn({ redirectUri: REDIRECT })).toThrow(
      /personal access token/,
    );
  });

  it("refreshes an expiring token once for concurrent dials, and keeps the new refresh token", async () => {
    const { bridge, registry, credentials } = setup(WITH_CLIENT);
    const saved = credentials.save({
      kind: CREDENTIAL_KIND_OAUTH,
      token: "old-access",
      refreshToken: "refresh-1",
      accessExpiresAt: inMinutes(0.5),
      clientId: "client-1",
      label: "DigitalOcean",
    });
    vi.mocked(refreshGrant).mockImplementationOnce(async () => ({
      ...GRANT,
      accessToken: "new-access",
      refreshToken: "refresh-2",
    }));
    await bridge.openTunnel({ sessionId: "sess_a" });
    const { getAccessToken } = registry.attach.mock.calls[0][0] as unknown as {
      getAccessToken: () => Promise<string | null>;
    };

    const tokens = await Promise.all([getAccessToken(), getAccessToken()]);

    expect(tokens).toEqual(["new-access", "new-access"]);
    expect(refreshGrant).toHaveBeenCalledTimes(1);
    expect(refreshGrant).toHaveBeenCalledWith({
      clientId: "client-1",
      refreshToken: "refresh-1",
    });
    expect(credentials.getGrant(saved.id)).toMatchObject({
      token: "new-access",
      refreshToken: "refresh-2",
    });
  });

  /** The tunnel registry's token getter for the first session opened. */
  async function dialToken(
    bridge: ReturnType<typeof setup>["bridge"],
    registry: ReturnType<typeof setup>["registry"],
  ) {
    await bridge.openTunnel({ sessionId: "sess_a" });
    const { getAccessToken } = registry.attach.mock.calls[0][0] as unknown as {
      getAccessToken: () => Promise<string | null>;
    };
    return getAccessToken;
  }

  const expiring = {
    kind: CREDENTIAL_KIND_OAUTH,
    token: "old-access",
    refreshToken: "refresh-1",
    clientId: "client-1",
    label: "DigitalOcean",
  };

  it("uses a refresh another Canvas process already made instead of spending the old refresh token", async () => {
    // Two processes sharing one store: each has its own cache and its own
    // refresh in flight, only the files in common.
    const dir = mkdtempSync(join(tmpdir(), "mars-shared-"));
    const storeFor = () =>
      createCredentialStore({
        userDataPath: dir,
        secrets: createEncryptedFileStore({ dir, secretKey: "k".repeat(64) }),
      });
    const first = storeFor();
    const saved = first.save({ ...expiring, accessExpiresAt: inMinutes(0.5) });
    const second = storeFor();
    const bridgeFor = (credentials: ReturnType<typeof storeFor>) => {
      const registry = fakeRegistry();
      const bridge = createMarsTunnelBridge({
        registry,
        api: fakeApi(),
        credentials,
        env: { ...WITH_CLIENT.env, MARS_API_BASE_URL: API_URL },
      });
      return { bridge, registry };
    };
    const a = bridgeFor(first);
    const b = bridgeFor(second);
    // One answer only: spending refresh-1 a second time is the bug.
    vi.mocked(refreshGrant).mockResolvedValueOnce({
      ...GRANT,
      accessToken: "new-access",
      refreshToken: "refresh-2",
    });

    expect(await (await dialToken(a.bridge, a.registry))()).toBe("new-access");
    expect(await (await dialToken(b.bridge, b.registry))()).toBe("new-access");

    expect(refreshGrant).toHaveBeenCalledTimes(1);
    expect(second.getGrant(saved.id)).toMatchObject({
      token: "new-access",
      refreshToken: "refresh-2",
    });
  });

  it("renews a sign-in the keychain refused to save, instead of losing it", async () => {
    const refusing = {
      ...createMemoryStore(),
      isPersistent: true,
      write: () => {
        throw new Error("macOS Keychain refused to save");
      },
    };
    const credentials = createCredentialStore({
      userDataPath: mkdtempSync(join(tmpdir(), "mars-unsaved-")),
      secrets: refusing,
      log: () => {},
    });
    const saved = credentials.save({
      ...expiring,
      accessExpiresAt: inMinutes(0.5),
    });
    expect(saved.isSaved).toBe(false);
    const registry = fakeRegistry();
    const bridge = createMarsTunnelBridge({
      registry,
      api: fakeApi(),
      credentials,
      env: { ...WITH_CLIENT.env, MARS_API_BASE_URL: API_URL },
    });
    vi.mocked(refreshGrant).mockResolvedValueOnce({
      ...GRANT,
      accessToken: "new-access",
      refreshToken: "refresh-2",
    });

    expect(await (await dialToken(bridge, registry))()).toBe("new-access");

    expect(refreshGrant).toHaveBeenCalledWith(
      expect.objectContaining({ refreshToken: "refresh-1" }),
    );
    expect(credentials.getGrant(saved.id)?.refreshToken).toBe("refresh-2");
  });

  it.each([
    {
      late: "fails",
      answer: () =>
        Promise.reject(
          new OAuthError("revoked", { status: 400, code: "invalid_grant" }),
        ),
    },
    {
      late: "succeeds",
      answer: async () => ({
        ...GRANT,
        accessToken: "refreshed-access",
        refreshToken: "refresh-2",
      }),
    },
  ])(
    "keeps a new sign-in for the team when a refresh already out $late",
    async ({ answer }) => {
      const { bridge, registry, credentials } = setup(WITH_CLIENT);
      const saved = credentials.save({
        ...expiring,
        accessExpiresAt: inMinutes(-1),
      });
      credentials.setTeam(saved.id, { uuid: "team-1", name: "My Team" });
      let release = () => {};
      vi.mocked(refreshGrant).mockImplementationOnce(
        () =>
          new Promise((resolve, reject) => {
            release = () => answer().then(resolve, reject);
          }),
      );
      const getAccessToken = await dialToken(bridge, registry);
      const dialing = getAccessToken();
      await vi.waitFor(() => expect(refreshGrant).toHaveBeenCalled());

      vi.mocked(exchangeCode).mockResolvedValueOnce({
        ...GRANT,
        accessToken: "signed-in-again",
        refreshToken: "refresh-new",
      });
      const { authorizeUrl } = bridge.beginSignIn({ redirectUri: REDIRECT });
      await bridge.completeSignIn(
        `${REDIRECT}?code=c&state=${stateOf(authorizeUrl)}`,
      );
      release();

      expect(await dialing).toBe("signed-in-again");
      expect(credentials.getGrant(saved.id)).toMatchObject({
        token: "signed-in-again",
        refreshToken: "refresh-new",
      });
    },
  );

  it("ends the grant a refresh got when a new sign-in replaced it meanwhile", async () => {
    const { bridge, registry, credentials } = setup(WITH_CLIENT);
    const saved = credentials.save({
      ...expiring,
      accessExpiresAt: inMinutes(-1),
    });
    credentials.setTeam(saved.id, { uuid: "team-1", name: "My Team" });
    let release = () => {};
    vi.mocked(refreshGrant).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          release = () =>
            resolve({
              ...GRANT,
              accessToken: "refreshed-access",
              refreshToken: "refresh-2",
            });
        }),
    );
    const dialing = (await dialToken(bridge, registry))();
    await vi.waitFor(() => expect(refreshGrant).toHaveBeenCalled());
    vi.mocked(exchangeCode).mockResolvedValueOnce({
      ...GRANT,
      accessToken: "signed-in-again",
      refreshToken: "refresh-new",
    });
    const { authorizeUrl } = bridge.beginSignIn({ redirectUri: REDIRECT });
    await bridge.completeSignIn(
      `${REDIRECT}?code=c&state=${stateOf(authorizeUrl)}`,
    );

    release();
    await dialing;

    expect(revokeToken).toHaveBeenCalledWith("refreshed-access");
  });

  it.each([
    { during: "while its code is exchanged", step: "exchange" },
    { during: "while its token is checked", step: "verify" },
  ])(
    "leaves everything as it was when a sign-in is cancelled $during, and ends the grant it got",
    async ({ step }) => {
      const { bridge, api, credentials } = setup(WITH_CLIENT);
      await bridge.savePat({ token: VALID_PAT });
      const before = bridge.getAuthState();
      let release = () => {};
      const held = new Promise<void>((resolve) => {
        release = resolve;
      });
      vi.mocked(exchangeCode).mockImplementationOnce(async () => {
        if (step === "exchange") await held;
        return { ...GRANT, accessToken: "cancelled-access" };
      });
      if (step === "verify") {
        vi.mocked(api.verifyAccess).mockImplementationOnce(() => held);
      }
      const { attemptId, authorizeUrl } = bridge.beginSignIn({
        redirectUri: REDIRECT,
      });
      const completing = bridge.completeSignIn(
        `${REDIRECT}?code=c&state=${stateOf(authorizeUrl)}`,
      );
      await vi.waitFor(() =>
        expect(
          step === "exchange" ? exchangeCode : api.verifyAccess,
        ).toHaveBeenCalled(),
      );

      bridge.cancelSignIn(attemptId);
      release();

      await expect(completing).rejects.toThrow(/cancelled/);
      expect(bridge.getAuthState()).toEqual(before);
      expect(credentials.list()).toHaveLength(1);
      expect(revokeToken).toHaveBeenCalledWith("cancelled-access");
    },
  );

  it("signs out by revoking a current access token, refreshing an expired one first", async () => {
    const { bridge, credentials } = setup(WITH_CLIENT);
    const saved = credentials.save({
      kind: CREDENTIAL_KIND_OAUTH,
      token: "old-access",
      refreshToken: "refresh-1",
      accessExpiresAt: inMinutes(-5),
      clientId: "client-1",
      label: "DigitalOcean",
    });
    vi.mocked(refreshGrant).mockResolvedValueOnce({
      ...GRANT,
      accessToken: "fresh-access",
    });

    await bridge.signOut(saved.id);

    expect(revokeToken).toHaveBeenCalledWith("fresh-access");
    expect(bridge.getAuthState().connections).toEqual([]);
  });

  it("asks for sign-in again when DigitalOcean no longer honours the refresh token", async () => {
    const { bridge, registry, credentials } = setup(WITH_CLIENT);
    credentials.save({
      kind: CREDENTIAL_KIND_OAUTH,
      token: "old-access",
      refreshToken: "refresh-1",
      accessExpiresAt: inMinutes(-1),
      clientId: "client-1",
      label: "DigitalOcean",
    });
    vi.mocked(refreshGrant).mockRejectedValueOnce(
      new OAuthError("Token is used", { status: 400, code: "invalid_grant" }),
    );
    await bridge.openTunnel({ sessionId: "sess_a" });
    const { getAccessToken } = registry.attach.mock.calls[0][0] as unknown as {
      getAccessToken: () => Promise<string | null>;
    };

    expect(await getAccessToken()).toBeNull();
    expect(bridge.getAuthState().active).toMatchObject({ isExpired: true });
  });
});
