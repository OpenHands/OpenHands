import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import ProviderConnectionsService, {
  isProviderConnectionsNotOnCloudError,
} from "./provider-connections-service";

const localBackend: Backend = {
  id: "local-test",
  name: "Local test backend",
  host: "http://localhost:3000",
  apiKey: "test-session-key",
  kind: "local",
};

const cloudBackend: Backend = {
  id: "cloud-test",
  name: "Cloud test backend",
  host: "https://cloud.example.com",
  apiKey: "cloud-key",
  kind: "cloud",
};

function mockFetchResponse(status: number, body?: unknown) {
  return vi.fn().mockResolvedValue({
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  });
}

function mockFetch204() {
  return vi.fn().mockResolvedValue({
    ok: true,
    status: 204,
    json: async () => null,
  });
}

const rawConnection = {
  id: "conn-1",
  provider: "openai",
  label: "Work",
  models: ["gpt-4o", "gpt-4o-mini"],
  created_at: 1700000000,
  last_validated_at: 1700000100,
  api_key_set: true,
};

describe("ProviderConnectionsService", () => {
  beforeEach(() => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
  });

  afterEach(() => {
    setActiveSelection(null);
    setRegisteredBackends([]);
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it("listConnections normalizes snake_case to camelCase and sends the session key", async () => {
    const fetchMock = mockFetchResponse(200, [rawConnection]);
    vi.stubGlobal("fetch", fetchMock);

    const result = await ProviderConnectionsService.listConnections();

    expect(result).toEqual([
      {
        id: "conn-1",
        provider: "openai",
        label: "Work",
        models: ["gpt-4o", "gpt-4o-mini"],
        createdAt: 1700000000,
        lastValidatedAt: 1700000100,
        apiKeySet: true,
      },
    ]);

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:3000/api/llm/connections");
    // GET is the fetch default; the service omits `method` for reads.
    expect(init.method ?? "GET").toBe("GET");
    expect((init.headers as Headers).get("X-Session-API-Key")).toBe(
      "test-session-key",
    );
  });

  it("createConnection POSTs provider + key (key never echoed back)", async () => {
    const fetchMock = mockFetchResponse(201, {
      ...rawConnection,
      models: [],
      last_validated_at: null,
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await ProviderConnectionsService.createConnection({
      provider: "openai",
      key: "sk-secret",
      label: "Work",
    });

    expect(result.apiKeySet).toBe(true);
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:3000/api/llm/connections");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({
      provider: "openai",
      key: "sk-secret",
      label: "Work",
    });
  });

  it("getConnection URL-encodes the id", async () => {
    const fetchMock = mockFetchResponse(200, rawConnection);
    vi.stubGlobal("fetch", fetchMock);

    await ProviderConnectionsService.getConnection("a/b c");

    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:3000/api/llm/connections/a%2Fb%20c");
  });

  it("updateConnection PATCHes only provided fields (key rotation)", async () => {
    const fetchMock = mockFetchResponse(200, rawConnection);
    vi.stubGlobal("fetch", fetchMock);

    await ProviderConnectionsService.updateConnection("conn-1", {
      key: "sk-new",
    });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:3000/api/llm/connections/conn-1");
    expect(init.method).toBe("PATCH");
    expect(JSON.parse(init.body as string)).toEqual({ key: "sk-new" });
  });

  it("updateConnection with models sends only models", async () => {
    const fetchMock = mockFetchResponse(200, rawConnection);
    vi.stubGlobal("fetch", fetchMock);

    await ProviderConnectionsService.updateConnection("conn-1", {
      models: ["gpt-4o"],
    });

    expect(JSON.parse(fetchMock.mock.calls[0][1].body as string)).toEqual({
      models: ["gpt-4o"],
    });
  });

  it("deleteConnection sends DELETE and returns affected profiles", async () => {
    const fetchMock = mockFetchResponse(200, {
      id: "conn-1",
      affected_profiles: ["work-gpt4o", "work-o3"],
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await ProviderConnectionsService.deleteConnection("conn-1");
    expect(result).toEqual({
      id: "conn-1",
      affectedProfiles: ["work-gpt4o", "work-o3"],
    });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("http://localhost:3000/api/llm/connections/conn-1");
    expect(init.method).toBe("DELETE");
  });

  it("deleteConnection tolerates a legacy empty (204) body", async () => {
    const fetchMock = mockFetch204();
    vi.stubGlobal("fetch", fetchMock);

    const result = await ProviderConnectionsService.deleteConnection("conn-1");
    expect(result).toEqual({ id: "conn-1", affectedProfiles: [] });
  });

  it("validateConnection POSTs to {id}/validate and normalizes", async () => {
    const fetchMock = mockFetchResponse(200, {
      id: "conn-1",
      provider: "openai",
      ok: true,
      verified: false,
      models: ["gpt-4o", "o3-mini"],
      error: null,
      validated_at: 1700000200,
    });
    vi.stubGlobal("fetch", fetchMock);

    const result =
      await ProviderConnectionsService.validateConnection("conn-1");

    expect(result).toEqual({
      id: "conn-1",
      provider: "openai",
      ok: true,
      verified: false,
      models: ["gpt-4o", "o3-mini"],
      error: null,
      validatedAt: 1700000200,
    });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(
      "http://localhost:3000/api/llm/connections/conn-1/validate",
    );
    expect(init.method).toBe("POST");
  });

  it("validateConnection forwards the live flag as a query param", async () => {
    const fetchMock = mockFetchResponse(200, {
      id: "conn-1",
      provider: "openai",
      ok: true,
      verified: true,
      models: ["gpt-4o"],
      error: null,
      validated_at: 1700000300,
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await ProviderConnectionsService.validateConnection(
      "conn-1",
      { live: true },
    );
    expect(result.verified).toBe(true);
    const [url] = fetchMock.mock.calls[0];
    expect(url).toBe(
      "http://localhost:3000/api/llm/connections/conn-1/validate?live=true",
    );
  });

  it("createProfileFromConnection POSTs to {id}/profiles with snake_case body", async () => {
    const fetchMock = mockFetchResponse(201, {
      profile_name: "gpt-4o",
      model: "gpt-4o",
      provider: "openai",
      connection_id: "conn-1",
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await ProviderConnectionsService.createProfileFromConnection(
      "conn-1",
      {
        profileName: "gpt-4o",
        model: "gpt-4o",
      },
    );

    expect(result).toEqual({
      profileName: "gpt-4o",
      model: "gpt-4o",
      provider: "openai",
      connectionId: "conn-1",
    });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(
      "http://localhost:3000/api/llm/connections/conn-1/profiles",
    );
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({
      profile_name: "gpt-4o",
      model: "gpt-4o",
    });
  });

  it("throws with the server detail on a non-2xx response", async () => {
    const fetchMock = mockFetchResponse(400, { detail: "provider unknown" });
    vi.stubGlobal("fetch", fetchMock);

    await expect(ProviderConnectionsService.listConnections()).rejects.toThrow(
      "provider unknown",
    );
  });

  it("is robust to a malformed connection object", async () => {
    const fetchMock = mockFetchResponse(200, [{ id: "x" }]);
    vi.stubGlobal("fetch", fetchMock);

    await expect(ProviderConnectionsService.listConnections()).rejects.toThrow(
      "missing id or provider",
    );
  });

  it("throws a typed error on cloud and never calls fetch", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });
    const fetchMock = mockFetchResponse(200, []);
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      ProviderConnectionsService.listConnections(),
    ).rejects.toSatisfy((e: unknown) =>
      isProviderConnectionsNotOnCloudError(e),
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
