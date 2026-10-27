// @vitest-environment node
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";

import {
  MARITIME_IPC,
  MISSING_ACCESS_KEY_CODE,
  corsOriginFor,
  createMaritimeBridge,
  readMaritimeConfig,
} from "../../scripts/maritime-bridge.mjs";
import { createMaritimeCredentialStore } from "../../scripts/maritime-credentials.mjs";
import { createMemoryStore } from "../../scripts/mars-secret-store.mjs";

const API_KEY = `mk_${"k".repeat(43)}`;
const WEB_URL = "https://agent-1.mtm.sh/";

type Agent = {
  id: string;
  name?: string;
  status: string;
  webUrl?: string | null;
};

function agent(status: string, overrides: Partial<Agent> = {}): Agent {
  return {
    id: "agent-1",
    name: "openhands-a",
    status,
    webUrl: WEB_URL,
    ...overrides,
  };
}

/** A Maritime API double whose getAgent walks through the given statuses. */
function fakeApi(statuses: string[] = ["active"]) {
  let index = 0;
  return {
    getAccountLimits: vi.fn(async () => ({
      plan: "starter",
      machines: 2,
      includedMachines: 20,
      awakeMachines: 1,
      maxAwakeMachines: 5,
      minMemMb: 512,
      maxMemMb: 16384,
    })),
    listAgents: vi.fn(async () => [
      agent("active"),
      agent("sleeping", { id: "agent-2" }),
    ]),
    getAgent: vi.fn(async () => {
      const status = statuses[Math.min(index, statuses.length - 1)];
      index += 1;
      return agent(status);
    }),
    createAgent: vi.fn(async () => agent("deploying")),
    startAgent: vi.fn(async () => agent("deploying")),
    sleepAgent: vi.fn(async () => agent("sleeping")),
    restartAgent: vi.fn(async () => agent("deploying")),
    deleteAgent: vi.fn(async () => null),
    setEnvVar: vi.fn(async () => ({})),
  };
}

function signedInBridge(api = fakeApi(), env: Record<string, string> = {}) {
  const credentials = createMaritimeCredentialStore();
  credentials.saveApiKey(API_KEY);
  const bridge = createMaritimeBridge({
    api: api as never,
    credentials,
    env: { MARITIME_IMAGE_TAG: "1.58.1", ...env },
    wait: async () => {},
  });
  return { bridge, api, credentials };
}

describe("readMaritimeConfig", () => {
  it("asks for the image tag of the agent-server version Canvas pins", () => {
    expect(readMaritimeConfig({}, { agentServerVersion: "1.58.1" })).toEqual({
      apiBaseUrl: "https://api.maritime.sh",
      imageName: "ghcr.io/maritime-sh/openhands-agent-server:1.58.1",
    });
  });

  it("lets env override the API, the image repository and the tag", () => {
    expect(
      readMaritimeConfig(
        {
          MARITIME_API_BASE_URL: "https://staging.maritime.test",
          MARITIME_IMAGE_REPOSITORY: "ghcr.io/acme/agent",
          MARITIME_IMAGE_TAG: "dev",
        },
        { agentServerVersion: "1.58.1" },
      ),
    ).toEqual({
      apiBaseUrl: "https://staging.maritime.test",
      imageName: "ghcr.io/acme/agent:dev",
    });
  });

  it("falls back to :latest where the shipped defaults are not beside the script (Docker)", () => {
    expect(readMaritimeConfig({}, {}).imageName).toBe(
      "ghcr.io/maritime-sh/openhands-agent-server:latest",
    );
  });
});

describe("corsOriginFor", () => {
  it("adds only non-loopback http(s) origins; the agent-server always allows loopback", () => {
    expect(corsOriginFor("https://canvas.example.com/path")).toBe(
      "https://canvas.example.com",
    );
    expect(corsOriginFor("http://localhost:8000")).toBeNull();
    expect(corsOriginFor("http://127.0.0.1:3001")).toBeNull();
    expect(corsOriginFor("http://[::1]:3001")).toBeNull();
    expect(corsOriginFor("http://canvas.localhost")).toBeNull();
    expect(corsOriginFor("file:///index.html")).toBeNull();
    expect(corsOriginFor("not a url")).toBeNull();
  });
});

describe("createMaritimeBridge", () => {
  it("keeps an API key only once Maritime accepts it, and never hands it back", async () => {
    const fetchImpl = vi.fn(async () => new Response("{}", { status: 200 }));
    const bridge = createMaritimeBridge({
      credentials: createMaritimeCredentialStore(),
      fetchImpl: fetchImpl as never,
      env: {},
    });

    const state = await bridge.saveApiKey({ apiKey: ` ${API_KEY} ` });

    expect(state.isSignedIn).toBe(true);
    expect(JSON.stringify(state)).not.toContain(API_KEY);
    const [url, init] = fetchImpl.mock.calls[0] as unknown as [
      string,
      RequestInit,
    ];
    expect(url).toBe("https://api.maritime.sh/api/v1/account/limits");
    expect(new Headers(init.headers).get("Authorization")).toBe(
      `Bearer ${API_KEY}`,
    );
  });

  it("does not keep a key Maritime rejects, and refuses a malformed one without calling out", async () => {
    const fetchImpl = vi.fn(
      async () => new Response('{"detail":"bad"}', { status: 401 }),
    );
    const bridge = createMaritimeBridge({
      credentials: createMaritimeCredentialStore(),
      fetchImpl: fetchImpl as never,
      env: {},
    });

    await expect(bridge.saveApiKey({ apiKey: API_KEY })).rejects.toThrow(
      /did not accept/,
    );
    expect(bridge.getAuthState().isSignedIn).toBe(false);

    await expect(bridge.saveApiKey({ apiKey: "dop_v1_x" })).rejects.toThrow(
      /mk_/,
    );
    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });

  it("creates an agent-server from the configured image, keeps its session key, and returns once it runs", async () => {
    const { bridge, api, credentials } = signedInBridge(
      fakeApi(["deploying", "deploying", "active"]),
    );

    const connection = await bridge.createAgent({
      name: "openhands-a",
      memMb: 4096,
      origin: "https://canvas.example.com",
    });

    const [[body]] = api.createAgent.mock.calls as unknown as [
      [
        {
          imageName: string;
          initialEnvVars: { key: string; value: string }[];
        },
      ],
    ];
    expect(body.imageName).toBe(
      "ghcr.io/maritime-sh/openhands-agent-server:1.58.1",
    );
    const env = Object.fromEntries(
      body.initialEnvVars.map((e) => [e.key, e.value]),
    );
    expect(env.OH_SESSION_API_KEYS_0).toMatch(/^[0-9a-f]{64}$/);
    expect(env.OH_SECRET_KEY).toMatch(/^[0-9a-f]{64}$/);
    expect(env.OH_SECRET_KEY).not.toBe(env.OH_SESSION_API_KEYS_0);
    expect(env.OH_ALLOW_CORS_ORIGINS_0).toBe("https://canvas.example.com");

    expect(connection).toEqual({
      agentId: "agent-1",
      name: "openhands-a",
      host: "https://agent-1.mtm.sh",
      sessionApiKey: env.OH_SESSION_API_KEYS_0,
    });
    expect(credentials.getSessionKey("agent-1")).toBe(
      env.OH_SESSION_API_KEYS_0,
    );
    expect(api.getAgent).toHaveBeenCalledTimes(3);
  });

  it("validates the name and memory before creating anything", async () => {
    const { bridge, api } = signedInBridge();
    await expect(
      bridge.createAgent({ name: "-bad name", memMb: 4096 }),
    ).rejects.toThrow(/letters, digits/);
    await expect(
      bridge.createAgent({ name: "ok", memMb: 999_999 }),
    ).rejects.toThrow(/memory/);
    expect(api.createAgent).not.toHaveBeenCalled();
  });

  it("wakes a sleeping agent once, then waits for it", async () => {
    const { bridge, api, credentials } = signedInBridge(
      fakeApi(["sleeping", "deploying", "active"]),
    );
    credentials.saveSessionKey("agent-1", "stored-key");

    const connection = await bridge.openAgent("agent-1");

    expect(api.startAgent).toHaveBeenCalledTimes(1);
    expect(connection.sessionApiKey).toBe("stored-key");
  });

  it("refuses to open an agent this host has no session key for, before calling Maritime", async () => {
    const { bridge, api } = signedInBridge();
    await expect(bridge.openAgent("agent-1")).rejects.toMatchObject({
      code: MISSING_ACCESS_KEY_CODE,
    });
    expect(api.getAgent).not.toHaveBeenCalled();
  });

  it("gives up on an agent Maritime marks as failed, and on one that never comes up", async () => {
    const failed = signedInBridge(fakeApi(["error"]));
    failed.credentials.saveSessionKey("agent-1", "k");
    await expect(failed.bridge.openAgent("agent-1")).rejects.toMatchObject({
      code: "agent_error",
    });

    let clock = 0;
    const api = fakeApi(["deploying"]);
    const credentials = createMaritimeCredentialStore();
    credentials.saveApiKey(API_KEY);
    credentials.saveSessionKey("agent-1", "k");
    const slow = createMaritimeBridge({
      api: api as never,
      credentials,
      env: {},
      now: () => clock,
      wait: async (ms: number) => {
        clock += ms;
      },
    });
    await expect(slow.openAgent("agent-1")).rejects.toMatchObject({
      code: "timeout",
    });
  });

  it("resets the access key by setting a new secret env var and restarting", async () => {
    const { bridge, api, credentials } = signedInBridge();

    const connection = await bridge.resetAccessKey("agent-1");

    expect(api.setEnvVar).toHaveBeenCalledWith(
      "agent-1",
      "OH_SESSION_API_KEYS_0",
      connection.sessionApiKey,
      { isSecret: true },
    );
    expect(api.restartAgent).toHaveBeenCalledWith("agent-1");
    expect(credentials.getSessionKey("agent-1")).toBe(connection.sessionApiKey);
  });

  it("lists agents with whether this host can open them, and forgets a key on delete", async () => {
    const { bridge, credentials } = signedInBridge();
    credentials.saveSessionKey("agent-1", "k");

    const agents = await bridge.listAgents();
    expect(
      agents.map((a: { id: string; hasAccessKey: boolean }) => [
        a.id,
        a.hasAccessKey,
      ]),
    ).toEqual([
      ["agent-1", true],
      ["agent-2", false],
    ]);

    await bridge.deleteAgent("agent-1");
    expect(credentials.getSessionKey("agent-1")).toBeNull();
  });

  it("treats a sleep that races another one as done once the agent is asleep", async () => {
    const api = fakeApi(["sleeping"]);
    api.sleepAgent.mockRejectedValueOnce(
      Object.assign(new Error("Internal server error"), { status: 500 }),
    );
    const { bridge } = signedInBridge(api);

    const result = await bridge.sleepAgent("agent-1");

    expect(result.status).toBe("sleeping");
  });

  it("still reports a failed sleep when the agent is not asleep", async () => {
    const api = fakeApi(["active"]);
    api.sleepAgent.mockRejectedValueOnce(
      Object.assign(new Error("Internal server error"), { status: 500 }),
    );
    const { bridge } = signedInBridge(api);

    await expect(bridge.sleepAgent("agent-1")).rejects.toThrow("Internal server error");
  });

  it("signs out without dropping agents' session keys", () => {
    const { bridge, credentials } = signedInBridge();
    credentials.saveSessionKey("agent-1", "k");

    expect(bridge.signOut().isSignedIn).toBe(false);
    expect(credentials.getSessionKey("agent-1")).toBe("k");
  });

  it("refuses every account call while signed out", async () => {
    const api = fakeApi();
    const bridge = createMaritimeBridge({
      api: api as never,
      credentials: createMaritimeCredentialStore(),
      env: {},
    });
    await expect(bridge.listAgents()).rejects.toMatchObject({ status: 401 });
    await expect(bridge.getAccount()).rejects.toMatchObject({ status: 401 });
    expect(api.listAgents).not.toHaveBeenCalled();
  });

  it("clamps the account's memory range to Maritime's self-serve range", async () => {
    const { bridge } = signedInBridge();
    await expect(bridge.getAccount()).resolves.toMatchObject({
      plan: "starter",
      minMemMb: 512,
      maxMemMb: 8192,
      maxAwakeMachines: 5,
    });
  });

  it("registers one IPC handler per bridge method", async () => {
    const { bridge } = signedInBridge();
    const handlers = new Map<string, (...args: unknown[]) => unknown>();
    bridge.registerIpc({
      handle: (channel: string, fn: (...args: unknown[]) => unknown) =>
        handlers.set(channel, fn),
    } as never);

    expect([...handlers.keys()].sort()).toEqual(
      Object.values(MARITIME_IPC).sort(),
    );
    const state = await handlers.get(MARITIME_IPC.getAuthState)!({});
    expect(state).toMatchObject({ isSignedIn: true });
    expect(JSON.stringify(state)).not.toContain(API_KEY);
  });
});

describe("createMaritimeCredentialStore", () => {
  it("keeps the record encrypted with safeStorage and reads it back after a restart", () => {
    const dir = mkdtempSync(join(tmpdir(), "maritime-creds-"));
    const safeStorage = {
      isEncryptionAvailable: () => true,
      encryptString: (value: string) =>
        Buffer.from(
          Buffer.from(value).toString("base64").split("").reverse().join(""),
        ),
      decryptString: (buffer: Buffer) =>
        Buffer.from(
          buffer.toString().split("").reverse().join(""),
          "base64",
        ).toString(),
    };
    const store = createMaritimeCredentialStore({
      userDataPath: dir,
      safeStorage,
    });
    store.saveApiKey(API_KEY);
    store.saveSessionKey("agent-1", "session");

    expect(
      readFileSync(join(dir, "maritime-credentials.json"), "utf8"),
    ).not.toContain(API_KEY);
    const reopened = createMaritimeCredentialStore({
      userDataPath: dir,
      safeStorage,
    });
    expect(reopened.getApiKey()).toBe(API_KEY);
    expect(reopened.getSessionKey("agent-1")).toBe("session");
    expect(reopened.isSaved).toBe(true);
  });

  it("uses the server's secret store when one is handed in", () => {
    const secrets = createMemoryStore();
    const store = createMaritimeCredentialStore({ secrets });
    store.saveApiKey(API_KEY);

    expect(createMaritimeCredentialStore({ secrets }).getApiKey()).toBe(
      API_KEY,
    );
  });

  it("says when nothing keeps the key past this run", () => {
    const store = createMaritimeCredentialStore();
    store.saveApiKey(API_KEY);
    expect(store.isSaved).toBe(false);
    expect(store.storage.kind).toBe("memory");
  });
});
