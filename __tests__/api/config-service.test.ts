import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { http, HttpResponse } from "msw";
import ConfigService from "#/api/config-service/config-service.api";
import { server } from "#/mocks/node";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";

const { cloudProxyMock } = vi.hoisted(() => ({ cloudProxyMock: vi.fn() }));

vi.mock("#/api/cloud/proxy", () => ({ callCloudProxy: cloudProxyMock }));

const cloudBackend: Backend = {
  id: "cloud",
  name: "Cloud",
  host: "https://cloud.example",
  apiKey: "token",
  kind: "cloud",
};

beforeEach(() => {
  cloudProxyMock.mockReset();
});

afterEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
  __resetActiveStoreForTests();
});

describe("ConfigService", () => {
  it("derives providers from llm endpoints", async () => {
    const page = await ConfigService.searchProviders({ limit: 10 });

    expect(page.next_page_id).toBeNull();
    expect(page.items.some((provider) => provider.name === "anthropic")).toBe(
      true,
    );
    expect(
      page.items.find((provider) => provider.name === "anthropic")?.verified,
    ).toBe(true);
  });

  it("derives provider models from llm endpoints", async () => {
    const page = await ConfigService.searchModels({
      provider__eq: "anthropic",
      limit: 20,
    });

    expect(page.next_page_id).toBeNull();
    expect(
      page.items.some((model) => model.name === "claude-opus-4-5-20251101"),
    ).toBe(true);
    expect(page.items.every((model) => model.provider === "anthropic")).toBe(
      true,
    );
  });

  it("includes verified providers absent from /api/llm/providers and keeps them within the limit", async () => {
    // Arrange: mirror the real local agent-server, where
    // /api/llm/providers comes from litellm (no "openhands"),
    // but /api/llm/models/verified has "openhands" as a key.
    const litellmOnlyProviders = Array.from(
      { length: 10 },
      (_, i) => `litellm_provider_${i}`,
    );
    server.use(
      http.get("/api/llm/providers", () =>
        HttpResponse.json({ providers: litellmOnlyProviders }),
      ),
      http.get("/api/llm/models/verified", () =>
        HttpResponse.json({
          models: {
            openhands: ["claude-opus-4-7", "gpt-5.5"],
            anthropic: ["claude-opus-4-5-20251101"],
          },
        }),
      ),
    );

    // Act: request fewer items than the litellm provider count to also
    // exercise the ordering fix (verified providers must come first so
    // they survive limitItems).
    const page = await ConfigService.searchProviders({ limit: 3 });

    // Assert
    const openhands = page.items.find((p) => p.name === "openhands");
    expect(openhands).toEqual({ name: "openhands", verified: true });
  });

  it("filters local models by query and verified status using supplied metadata", async () => {
    const verified = await ConfigService.searchModels(
      {
        provider__eq: "anthropic",
        query: "OPUS",
        verified__eq: true,
        limit: 1,
      },
      { anthropic: ["claude-opus-4-5-20251101"] },
    );
    expect(verified.items).toEqual([
      {
        provider: "anthropic",
        name: "claude-opus-4-5-20251101",
        verified: true,
        free: false,
        default: false,
      },
    ]);

    const unverified = await ConfigService.searchModels(
      { provider__eq: "anthropic", verified__eq: false },
      {},
    );
    expect(unverified.items.length).toBeGreaterThan(0);
    expect(unverified.items.every((model) => !model.verified)).toBe(true);

    const withoutProvider = await ConfigService.searchModels({}, {});
    expect(withoutProvider.items).toEqual([]);
  });

  it("normalizes provider models and honors verified metadata precedence", async () => {
    server.use(
      http.get("/api/llm/models", () =>
        HttpResponse.json({
          models: [
            "openai/foreign-model",
            "anthropic/verified-model",
            "anthropic/",
            "anthropic/unverified-a",
            "anthropic/unverified-b",
          ],
        }),
      ),
      http.get("/api/llm/models/verified", () =>
        HttpResponse.json({
          models: { anthropic: ["server-verified"] },
        }),
      ),
    );
    const supplied = { anthropic: ["verified-model"] };

    await expect(
      ConfigService.searchModels(
        { provider__eq: "anthropic", limit: 2 },
        supplied,
      ),
    ).resolves.toEqual({
      items: [
        {
          provider: "anthropic",
          name: "verified-model",
          verified: true,
          free: false,
          default: false,
        },
        {
          provider: "anthropic",
          name: "unverified-a",
          verified: false,
          free: false,
          default: false,
        },
      ],
      next_page_id: null,
    });

    await expect(
      ConfigService.searchModels(
        { provider__eq: "anthropic", verified__eq: true },
        supplied,
      ),
    ).resolves.toEqual({
      items: [
        {
          provider: "anthropic",
          name: "verified-model",
          verified: true,
          free: false,
          default: false,
        },
      ],
      next_page_id: null,
    });

    await expect(
      ConfigService.searchModels({
        provider__eq: "anthropic",
        verified__eq: true,
      }),
    ).resolves.toEqual({
      items: [
        {
          provider: "anthropic",
          name: "server-verified",
          verified: true,
          free: false,
          default: false,
        },
      ],
      next_page_id: null,
    });
  });

  it("filters local providers and does not limit for non-positive values", async () => {
    const page = await ConfigService.searchProviders(
      { query: "anth", verified__eq: false, limit: -1 },
      { openhands: ["model"] },
    );
    expect(page.items).toEqual([{ name: "anthropic", verified: false }]);
  });

  it("handles null model/provider metadata from the local SDK", async () => {
    server.use(
      http.get("/api/llm/models", () => HttpResponse.json({ models: null })),
      http.get("/api/llm/providers", () =>
        HttpResponse.json({ providers: null }),
      ),
      http.get("/api/llm/models/verified", () =>
        HttpResponse.json({ models: null }),
      ),
    );

    await expect(
      ConfigService.searchModels({ provider__eq: "missing" }),
    ).resolves.toEqual({ items: [], next_page_id: null });
    await expect(ConfigService.searchProviders()).resolves.toEqual({
      items: [],
      next_page_id: null,
    });
  });

  it("forwards every model search parameter to the cloud API", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });
    const response = { items: [], next_page_id: "next" };
    cloudProxyMock.mockResolvedValue(response);

    await expect(
      ConfigService.searchModels(
        {
          page_id: "page 1",
          limit: 5,
          query: "claude",
          verified__eq: false,
          provider__eq: "anthropic",
        },
        { ignored: ["model"] },
      ),
    ).resolves.toBe(response);
    expect(cloudProxyMock).toHaveBeenCalledWith({
      backend: cloudBackend,
      method: "GET",
      path: "/api/v1/config/models/search?page_id=page+1&limit=5&query=claude&verified__eq=false&provider__eq=anthropic",
    });
  });

  it("uses the bare cloud provider search path when params are absent", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });
    const response = { items: [], next_page_id: null };
    cloudProxyMock.mockResolvedValue(response);

    await expect(ConfigService.searchProviders()).resolves.toBe(response);
    expect(cloudProxyMock).toHaveBeenCalledWith({
      backend: cloudBackend,
      method: "GET",
      path: "/api/v1/config/providers/search",
    });
  });

  it("forwards every provider search parameter to the cloud API", async () => {
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });
    const response = { items: [], next_page_id: "next" };
    cloudProxyMock.mockResolvedValue(response);

    await expect(
      ConfigService.searchProviders({
        page_id: "page 1",
        limit: 5,
        query: "anthropic",
        verified__eq: false,
      }),
    ).resolves.toBe(response);
    expect(cloudProxyMock).toHaveBeenCalledWith({
      backend: cloudBackend,
      method: "GET",
      path: "/api/v1/config/providers/search?page_id=page+1&limit=5&query=anthropic&verified__eq=false",
    });
  });
});
