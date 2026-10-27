// @vitest-environment node
import { describe, expect, it, vi } from "vitest";

import {
  MARITIME_PUBLIC_PORT,
  OPENHANDS_EXTERNAL_ID,
  buildImageName,
  buildOpenHandsAgentRequest,
  createMaritimeApiClient,
  isValidApiKeyFormat,
  vcpusForMemory,
} from "../../scripts/maritime-api.mjs";

const BASE_URL = "https://api.maritime.test";
const API_KEY = `mk_${"a".repeat(43)}`;

function jsonResponse(status: number, body: unknown) {
  return new Response(body === undefined ? null : JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function clientWith(fetchImpl: typeof fetch, apiKey: string | null = API_KEY) {
  return createMaritimeApiClient({
    baseUrl: `${BASE_URL}/`,
    getApiKey: () => apiKey,
    fetchImpl,
  });
}

describe("isValidApiKeyFormat", () => {
  it("accepts mk_ keys and rejects anything else before it is sent", () => {
    expect(isValidApiKeyFormat(` ${API_KEY} `)).toBe(true);
    expect(isValidApiKeyFormat("mk_short")).toBe(false);
    expect(isValidApiKeyFormat(`dop_v1_${"a".repeat(64)}`)).toBe(false);
    expect(isValidApiKeyFormat(undefined)).toBe(false);
  });
});

describe("buildImageName", () => {
  it("joins the repository and the pinned agent-server version", () => {
    expect(buildImageName("ghcr.io/maritime-sh/x", "1.58.1")).toBe(
      "ghcr.io/maritime-sh/x:1.58.1",
    );
  });

  it("refuses a tag that could smuggle anything into the image reference", () => {
    expect(() => buildImageName("repo", "1.0 --evil")).toThrow(/tag/);
    expect(() => buildImageName("repo", "")).toThrow(/tag/);
  });
});

describe("buildOpenHandsAgentRequest", () => {
  it("creates a public custom agent on Maritime's routed port with the agent-server secrets as secret env vars", () => {
    const body = buildOpenHandsAgentRequest({
      name: "openhands-abc",
      imageName: "ghcr.io/maritime-sh/openhands-agent-server:1.58.1",
      memMb: 4096,
      sessionApiKey: "session-secret",
      secretKey: "settings-secret",
      corsOrigins: ["https://canvas.example.com"],
    });

    expect(body).toMatchObject({
      name: "openhands-abc",
      externalId: OPENHANDS_EXTERNAL_ID,
      framework: "custom",
      imageName: "ghcr.io/maritime-sh/openhands-agent-server:1.58.1",
      exposedPort: MARITIME_PUBLIC_PORT,
      publicWeb: true,
      memMb: 4096,
      vcpus: 2,
    });
    expect(body.idleTtlSeconds).toBeGreaterThan(0);
    expect(body.initialEnvVars).toEqual([
      { key: "OH_SESSION_API_KEYS_0", value: "session-secret", isSecret: true },
      { key: "OH_SECRET_KEY", value: "settings-secret", isSecret: true },
      {
        key: "OH_ALLOW_CORS_ORIGINS_0",
        value: "https://canvas.example.com",
        isSecret: false,
      },
    ]);
    // Maritime injects PORT itself; setting it here would fight the forwarder.
    expect(
      body.initialEnvVars.map((e: { key: string }) => e.key),
    ).not.toContain("PORT");
  });
});

describe("vcpusForMemory", () => {
  it("gives one vCPU per 2 GB, between one and four", () => {
    expect([512, 2048, 4096, 8192].map(vcpusForMemory)).toEqual([1, 1, 2, 4]);
  });
});

describe("createMaritimeApiClient", () => {
  it("sends the key as a bearer token and filters the list by the Canvas tag", async () => {
    const fetchImpl = vi.fn(async () =>
      jsonResponse(200, [{ id: "a1", name: "one" }]),
    );
    const agents = await clientWith(fetchImpl).listAgents();

    expect(agents).toEqual([{ id: "a1", name: "one" }]);
    const [url, init] = fetchImpl.mock.calls[0] as unknown as [
      string,
      RequestInit,
    ];
    expect(url).toBe(
      `${BASE_URL}/api/agents?externalId=${OPENHANDS_EXTERNAL_ID}`,
    );
    expect(new Headers(init.headers).get("Authorization")).toBe(
      `Bearer ${API_KEY}`,
    );
  });

  it("refuses to call Maritime with no key", async () => {
    const fetchImpl = vi.fn();
    await expect(
      clientWith(fetchImpl as never, null).getAccountLimits(),
    ).rejects.toMatchObject({ status: 401 });
    expect(fetchImpl).not.toHaveBeenCalled();
  });

  it("turns Maritime's status codes into messages that survive Electron IPC", async () => {
    const cases: [number, unknown, RegExp][] = [
      [401, { detail: "Invalid token" }, /did not accept this API key/],
      [403, { detail: "Missing scope" }, /provision, deploy, secrets and manage/],
      [
        402,
        { detail: "Free plan holds 3 machines", code: "seat_limit" },
        /Free plan holds 3 machines/,
      ],
      [429, { detail: "5 awake at most" }, /limit reached: 5 awake at most/],
      [409, { detail: 'You already have an agent named "x".' }, /already have/],
      [422, { detail: [{ msg: "memMb too large" }] }, /memMb too large/],
    ];
    for (const [status, body, message] of cases) {
      const fetchImpl = vi.fn(async () => jsonResponse(status, body));
      // eslint-disable-next-line no-await-in-loop
      await expect(clientWith(fetchImpl).createAgent({})).rejects.toMatchObject(
        { status, message: expect.stringMatching(message) },
      );
    }
  });

  it("keeps the error code Maritime sends", async () => {
    const fetchImpl = vi.fn(async () =>
      jsonResponse(402, { detail: "Upgrade", code: "seat_limit" }),
    );
    await expect(clientWith(fetchImpl).createAgent({})).rejects.toMatchObject({
      code: "seat_limit",
    });
  });

  it("answers null for 204 and posts env vars to the upsert endpoint", async () => {
    const fetchImpl = vi.fn(async () => new Response(null, { status: 204 }));
    const client = clientWith(fetchImpl);

    await expect(client.deleteAgent("a/1")).resolves.toBeNull();
    await client.setEnvVar("a1", "OH_SESSION_API_KEYS_0", "new-key");

    const deleteCall = fetchImpl.mock.calls[0] as unknown as [
      string,
      RequestInit,
    ];
    expect(deleteCall[0]).toBe(`${BASE_URL}/api/agents/a%2F1`);
    expect(deleteCall[1].method).toBe("DELETE");
    const envCall = fetchImpl.mock.calls[1] as unknown as [string, RequestInit];
    expect(envCall[0]).toBe(`${BASE_URL}/api/agents/a1/env`);
    expect(JSON.parse(String(envCall[1].body))).toEqual({
      key: "OH_SESSION_API_KEYS_0",
      value: "new-key",
      isSecret: true,
    });
  });

  it("reports an unreachable API without leaking the key", async () => {
    const fetchImpl = vi.fn(async () => {
      throw new TypeError(`fetch failed for ${API_KEY}`);
    });
    const error = await clientWith(fetchImpl)
      .getAgent("a1")
      .catch((e: Error) => e);
    expect(error).toMatchObject({ message: "Could not reach Maritime." });
  });
});
