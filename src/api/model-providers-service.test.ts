import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import ModelProvidersService, {
  isModelProvidersNotOnCloudError,
} from "./model-providers-service";

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

const rawProvider = {
  id: "prov-1",
  kind: "openai",
  display_name: "OpenAI",
  base_url: "https://api.openai.com/v1",
  wire_api: "chat",
  custom_headers: { "X-Org": "eng" },
  models: [
    { name: "gpt-5.6-luna", wire_api: null },
    { name: "gpt-5.6-sol", wire_api: "responses" },
  ],
  created_at: 1700000000,
  updated_at: 1700000100,
  api_key_set: true,
};

describe("ModelProvidersService", () => {
  beforeEach(() => {
    setRegisteredBackends([localBackend]);
    setActiveSelection({ backendId: localBackend.id });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("lists providers and normalizes snake_case to camelCase", async () => {
    const fetchMock = mockFetchResponse(200, [rawProvider]);
    vi.stubGlobal("fetch", fetchMock);

    const providers = await ModelProvidersService.listProviders();

    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:3000/api/llm/model-providers",
      expect.objectContaining({ headers: expect.any(Headers) }),
    );
    expect(providers).toHaveLength(1);
    expect(providers[0]).toMatchObject({
      id: "prov-1",
      kind: "openai",
      displayName: "OpenAI",
      baseUrl: "https://api.openai.com/v1",
      wireApi: "chat",
      customHeaders: { "X-Org": "eng" },
      apiKeySet: true,
    });
    expect(providers[0].models).toEqual([
      { name: "gpt-5.6-luna", wireApi: null },
      { name: "gpt-5.6-sol", wireApi: "responses" },
    ]);
  });

  it("creates a provider with a mapped request body", async () => {
    const fetchMock = mockFetchResponse(201, rawProvider);
    vi.stubGlobal("fetch", fetchMock);

    await ModelProvidersService.createProvider({
      kind: "openai",
      displayName: "OpenAI",
      key: "sk-secret",
      baseUrl: "https://api.openai.com/v1",
      wireApi: "chat",
      customHeaders: { "X-Org": "eng" },
    });

    const [, init] = fetchMock.mock.calls[0];
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({
      kind: "openai",
      display_name: "OpenAI",
      key: "sk-secret",
      base_url: "https://api.openai.com/v1",
      wire_api: "chat",
      custom_headers: { "X-Org": "eng" },
    });
  });

  it("adds a model under a provider", async () => {
    const fetchMock = mockFetchResponse(201, rawProvider);
    vi.stubGlobal("fetch", fetchMock);

    await ModelProvidersService.addModel("prov-1", { name: "gpt-5.6-terra" });

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(
      "http://localhost:3000/api/llm/model-providers/prov-1/models",
    );
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({
      name: "gpt-5.6-terra",
      wire_api: null,
    });
  });

  it("removes a model, URL-encoding the model name", async () => {
    const fetchMock = mockFetchResponse(200, rawProvider);
    vi.stubGlobal("fetch", fetchMock);

    await ModelProvidersService.removeModel("prov-1", "org/model:v1");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(
      "http://localhost:3000/api/llm/model-providers/prov-1/models/org%2Fmodel%3Av1",
    );
    expect(init.method).toBe("DELETE");
  });

  it("only sends a key on update when one is provided", async () => {
    const fetchMock = mockFetchResponse(200, rawProvider);
    vi.stubGlobal("fetch", fetchMock);

    await ModelProvidersService.updateProvider("prov-1", {
      displayName: "Renamed",
    });

    const [, init] = fetchMock.mock.calls[0];
    const body = JSON.parse(init.body as string);
    expect(body).toEqual({ display_name: "Renamed" });
    expect(body).not.toHaveProperty("key");
  });

  it("normalizes the test-probe response", async () => {
    const fetchMock = mockFetchResponse(200, {
      id: "prov-1",
      ok: true,
      verified: true,
      suggested_models: ["gpt-5.6-luna"],
      error: null,
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await ModelProvidersService.testProvider("prov-1");
    expect(result).toEqual({
      id: "prov-1",
      ok: true,
      verified: true,
      suggestedModels: ["gpt-5.6-luna"],
      error: null,
    });
  });

  it("surfaces the server error detail on failure", async () => {
    const fetchMock = mockFetchResponse(400, { detail: "bad request" });
    vi.stubGlobal("fetch", fetchMock);

    await expect(ModelProvidersService.listProviders()).rejects.toThrow(
      "bad request",
    );
  });

  it("throws a cloud-not-supported error on cloud backends", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });
    const fetchMock = mockFetchResponse(200, []);
    vi.stubGlobal("fetch", fetchMock);

    await expect(ModelProvidersService.listProviders()).rejects.toSatisfy(
      isModelProvidersNotOnCloudError,
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
