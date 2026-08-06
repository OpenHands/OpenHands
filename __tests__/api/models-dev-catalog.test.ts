import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ACPModelOption } from "#/constants/acp-providers";
import {
  __resetModelsDevCatalogCacheForTests,
  fetchModelsDevCatalog,
  getCatalogModelsForAcpServer,
  getEffortValuesForModel,
  getModelsDevProviderKey,
  mergeModelOptions,
  MODELS_DEV_CATALOG_STORAGE_KEY,
  MODELS_DEV_CATALOG_TTL_MS,
  MODELS_DEV_PROVIDER_BY_ACP_SERVER,
  trimModelsDevCatalog,
  type ModelsDevCatalog,
  type ModelsDevModel,
} from "#/api/models-dev-catalog";

// A small, realistic slice of the real models.dev api.json shape — enough
// to exercise trimming (label/description/efforts/contextLimit) without
// pulling in the full ~3.5MB payload.
const RAW_FIXTURE = {
  anthropic: {
    id: "anthropic",
    name: "Anthropic",
    models: {
      "claude-opus-4-6": {
        id: "claude-opus-4-6",
        name: "Claude Opus 4.6",
        description: "Anthropic's most capable model.",
        reasoning: true,
        reasoning_options: [
          { type: "effort", values: ["low", "medium", "high"] },
        ],
        limit: { context: 200000, output: 32000 },
        release_date: "2026-01-01",
        family: "claude",
      },
      "claude-haiku-4-6": {
        id: "claude-haiku-4-6",
        name: "Claude Haiku 4.6",
        limit: { context: 200000, output: 8192 },
      },
    },
  },
  openai: {
    id: "openai",
    name: "OpenAI",
    models: {
      "gpt-5.1": {
        id: "gpt-5.1",
        name: "GPT-5.1",
        reasoning: true,
        // A non-"effort" reasoning option should NOT populate `efforts`.
        reasoning_options: [{ type: "budget_tokens", min: 1024 }],
        limit: { context: 400000, output: 128000 },
      },
    },
  },
  google: {
    id: "google",
    name: "Google",
    models: {
      "gemini-2.5-pro": {
        id: "gemini-2.5-pro",
        name: "Gemini 2.5 Pro",
        limit: { context: 1000000, output: 65536 },
      },
    },
  },
};

const TRIMMED_FIXTURE: ModelsDevCatalog = {
  anthropic: {
    id: "anthropic",
    name: "Anthropic",
    models: [
      {
        id: "claude-opus-4-6",
        label: "Claude Opus 4.6",
        description: "Anthropic's most capable model.",
        efforts: ["low", "medium", "high"],
        contextLimit: 200000,
      },
      {
        id: "claude-haiku-4-6",
        label: "Claude Haiku 4.6",
        contextLimit: 200000,
      },
    ],
  },
  openai: {
    id: "openai",
    name: "OpenAI",
    models: [
      {
        id: "gpt-5.1",
        label: "GPT-5.1",
        contextLimit: 400000,
      },
    ],
  },
  google: {
    id: "google",
    name: "Google",
    models: [
      {
        id: "gemini-2.5-pro",
        label: "Gemini 2.5 Pro",
        contextLimit: 1000000,
      },
    ],
  },
};

function stubFetch(impl: (...args: unknown[]) => unknown) {
  const fetchMock = vi.fn(impl);
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function jsonResponse(
  body: unknown,
  init: { etag?: string | null; ok?: boolean; status?: number } = {},
) {
  const { etag = null, ok = true, status = 200 } = init;
  return {
    ok,
    status,
    headers: { get: (name: string) => (name === "etag" ? etag : null) },
    json: async () => body,
  };
}

beforeEach(() => {
  window.localStorage.clear();
  __resetModelsDevCatalogCacheForTests();
});

afterEach(() => {
  window.localStorage.clear();
  __resetModelsDevCatalogCacheForTests();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("provider mapping", () => {
  it("maps the built-in ACP servers to their models.dev provider id", () => {
    expect(MODELS_DEV_PROVIDER_BY_ACP_SERVER).toEqual({
      "claude-code": "anthropic",
      codex: "openai",
      "gemini-cli": "google",
    });
    expect(getModelsDevProviderKey("claude-code")).toBe("anthropic");
    expect(getModelsDevProviderKey("codex")).toBe("openai");
    expect(getModelsDevProviderKey("gemini-cli")).toBe("google");
  });

  it("returns null for custom or unknown ACP servers", () => {
    expect(getModelsDevProviderKey("custom")).toBeNull();
    expect(getModelsDevProviderKey("some-future-provider")).toBeNull();
  });
});

describe("trimModelsDevCatalog", () => {
  it("trims the raw payload to the compact internal shape, extracting effort values", () => {
    expect(trimModelsDevCatalog(RAW_FIXTURE)).toEqual(TRIMMED_FIXTURE);
  });

  it("returns null for a non-object payload", () => {
    expect(trimModelsDevCatalog(null)).toBeNull();
    expect(trimModelsDevCatalog("not an object")).toBeNull();
    expect(trimModelsDevCatalog(42)).toBeNull();
  });

  it("skips malformed provider/model entries instead of failing the whole parse", () => {
    const result = trimModelsDevCatalog({
      anthropic: { id: "anthropic", name: "Anthropic", models: null },
      broken: "not an object",
      openai: {
        id: "openai",
        name: "OpenAI",
        models: {
          "gpt-5.1": { id: "gpt-5.1", name: "GPT-5.1" },
          "bad-model": "also not an object",
        },
      },
    });

    expect(result).toEqual({
      anthropic: { id: "anthropic", name: "Anthropic", models: [] },
      openai: {
        id: "openai",
        name: "OpenAI",
        models: [{ id: "gpt-5.1", label: "GPT-5.1" }],
      },
    });
  });
});

describe("getCatalogModelsForAcpServer", () => {
  it("returns the trimmed models for the mapped provider", () => {
    expect(getCatalogModelsForAcpServer(TRIMMED_FIXTURE, "claude-code")).toBe(
      TRIMMED_FIXTURE.anthropic.models,
    );
  });

  it("returns [] when the catalog is missing", () => {
    expect(getCatalogModelsForAcpServer(null, "claude-code")).toEqual([]);
    expect(getCatalogModelsForAcpServer(undefined, "claude-code")).toEqual([]);
  });

  it("returns [] for a custom server with no explicit providerKey", () => {
    expect(getCatalogModelsForAcpServer(TRIMMED_FIXTURE, "custom")).toEqual([]);
  });

  it("honors an explicit providerKey override for custom servers", () => {
    expect(
      getCatalogModelsForAcpServer(TRIMMED_FIXTURE, "custom", {
        providerKey: "openai",
      }),
    ).toBe(TRIMMED_FIXTURE.openai.models);
  });

  it("returns [] when the resolved provider key isn't in the catalog", () => {
    expect(
      getCatalogModelsForAcpServer(TRIMMED_FIXTURE, "custom", {
        providerKey: "unknown-provider",
      }),
    ).toEqual([]);
  });
});

describe("mergeModelOptions", () => {
  const curated: ACPModelOption[] = [
    { id: "claude-opus-4-6", label: "Claude Opus 4.6 (Curated)" },
    { id: "claude-sonnet-4-6", label: "Claude Sonnet 4.6" },
  ];

  it("puts curated entries first (preserving order) tagged source: curated", () => {
    const merged = mergeModelOptions(curated, []);
    expect(merged).toEqual([
      {
        id: "claude-opus-4-6",
        label: "Claude Opus 4.6 (Curated)",
        source: "curated",
      },
      {
        id: "claude-sonnet-4-6",
        label: "Claude Sonnet 4.6",
        source: "curated",
      },
    ]);
  });

  it("appends catalog models not already curated, tagged source: models.dev", () => {
    const catalogModels: ModelsDevModel[] = TRIMMED_FIXTURE.anthropic.models;
    const merged = mergeModelOptions(curated, catalogModels);

    expect(merged).toEqual([
      {
        id: "claude-opus-4-6",
        label: "Claude Opus 4.6 (Curated)",
        source: "curated",
      },
      {
        id: "claude-sonnet-4-6",
        label: "Claude Sonnet 4.6",
        source: "curated",
      },
      {
        id: "claude-haiku-4-6",
        label: "Claude Haiku 4.6",
        source: "models.dev",
        description: undefined,
        efforts: undefined,
      },
    ]);
  });

  it("dedupes by exact id match — a catalog model matching a curated id is dropped, not duplicated", () => {
    const catalogModels: ModelsDevModel[] = [
      {
        id: "claude-opus-4-6",
        label: "Claude Opus 4.6",
        description: "duplicate of curated",
      },
    ];
    const merged = mergeModelOptions(curated, catalogModels);

    expect(merged).toHaveLength(2);
    expect(merged.filter((m) => m.id === "claude-opus-4-6")).toHaveLength(1);
    expect(merged[0]).toMatchObject({
      id: "claude-opus-4-6",
      label: "Claude Opus 4.6 (Curated)",
      source: "curated",
    });
  });

  it("carries description and efforts through for catalog-only entries", () => {
    const merged = mergeModelOptions([], [TRIMMED_FIXTURE.anthropic.models[0]]);
    expect(merged[0]).toMatchObject({
      id: "claude-opus-4-6",
      description: "Anthropic's most capable model.",
      efforts: ["low", "medium", "high"],
      source: "models.dev",
    });
  });
});

describe("getEffortValuesForModel", () => {
  it("returns the effort values for a model that has them", () => {
    expect(
      getEffortValuesForModel(TRIMMED_FIXTURE, "anthropic", "claude-opus-4-6"),
    ).toEqual(["low", "medium", "high"]);
  });

  it("returns null when the model has no effort metadata", () => {
    expect(
      getEffortValuesForModel(TRIMMED_FIXTURE, "anthropic", "claude-haiku-4-6"),
    ).toBeNull();
  });

  it("returns null for an unknown provider, model, or missing catalog", () => {
    expect(
      getEffortValuesForModel(null, "anthropic", "claude-opus-4-6"),
    ).toBeNull();
    expect(
      getEffortValuesForModel(TRIMMED_FIXTURE, "unknown", "claude-opus-4-6"),
    ).toBeNull();
    expect(
      getEffortValuesForModel(TRIMMED_FIXTURE, "anthropic", "unknown-model"),
    ).toBeNull();
  });
});

describe("fetchModelsDevCatalog", () => {
  it("fetches and caches the catalog when there is nothing cached yet", async () => {
    const fetchMock = stubFetch(() =>
      jsonResponse(RAW_FIXTURE, { etag: '"v1"' }),
    );

    const result = await fetchModelsDevCatalog();

    expect(result).toEqual(TRIMMED_FIXTURE);
    expect(fetchMock).toHaveBeenCalledWith(
      "https://models.dev/api.json",
      expect.objectContaining({ headers: {} }),
    );

    const stored = JSON.parse(
      window.localStorage.getItem(MODELS_DEV_CATALOG_STORAGE_KEY) ?? "null",
    );
    expect(stored.etag).toBe('"v1"');
    expect(stored.catalog).toEqual(TRIMMED_FIXTURE);
  });

  it("serves a fresh cache without touching the network", async () => {
    window.localStorage.setItem(
      MODELS_DEV_CATALOG_STORAGE_KEY,
      JSON.stringify({
        etag: '"cached-etag"',
        fetchedAt: Date.now(),
        catalog: TRIMMED_FIXTURE,
      }),
    );
    const fetchMock = stubFetch(() => {
      throw new Error("should not be called for a fresh cache");
    });

    const result = await fetchModelsDevCatalog();

    expect(result).toEqual(TRIMMED_FIXTURE);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("revalidates a stale cache with If-None-Match and refreshes fetchedAt on 304", async () => {
    const staleFetchedAt = Date.now() - MODELS_DEV_CATALOG_TTL_MS - 1000;
    window.localStorage.setItem(
      MODELS_DEV_CATALOG_STORAGE_KEY,
      JSON.stringify({
        etag: '"cached-etag"',
        fetchedAt: staleFetchedAt,
        catalog: TRIMMED_FIXTURE,
      }),
    );
    const fetchMock = stubFetch(() =>
      jsonResponse(undefined, { ok: false, status: 304 }),
    );

    const result = await fetchModelsDevCatalog();

    expect(result).toEqual(TRIMMED_FIXTURE);
    expect(fetchMock).toHaveBeenCalledWith(
      "https://models.dev/api.json",
      expect.objectContaining({
        headers: { "If-None-Match": '"cached-etag"' },
      }),
    );

    const stored = JSON.parse(
      window.localStorage.getItem(MODELS_DEV_CATALOG_STORAGE_KEY) ?? "null",
    );
    expect(stored.fetchedAt).toBeGreaterThan(staleFetchedAt);
    expect(stored.etag).toBe('"cached-etag"');
  });

  it("replaces a stale cache with fresh data on 200", async () => {
    const staleFetchedAt = Date.now() - MODELS_DEV_CATALOG_TTL_MS - 1000;
    window.localStorage.setItem(
      MODELS_DEV_CATALOG_STORAGE_KEY,
      JSON.stringify({
        etag: '"old-etag"',
        fetchedAt: staleFetchedAt,
        catalog: TRIMMED_FIXTURE,
      }),
    );
    const updatedRaw = {
      anthropic: {
        id: "anthropic",
        name: "Anthropic",
        models: {
          "claude-opus-5": { id: "claude-opus-5", name: "Claude Opus 5" },
        },
      },
    };
    stubFetch(() => jsonResponse(updatedRaw, { etag: '"new-etag"' }));

    const result = await fetchModelsDevCatalog();

    expect(result).toEqual({
      anthropic: {
        id: "anthropic",
        name: "Anthropic",
        models: [{ id: "claude-opus-5", label: "Claude Opus 5" }],
      },
    });

    const stored = JSON.parse(
      window.localStorage.getItem(MODELS_DEV_CATALOG_STORAGE_KEY) ?? "null",
    );
    expect(stored.etag).toBe('"new-etag"');
  });

  it("serves the stale cache when the network request fails outright", async () => {
    const staleFetchedAt = Date.now() - MODELS_DEV_CATALOG_TTL_MS - 1000;
    window.localStorage.setItem(
      MODELS_DEV_CATALOG_STORAGE_KEY,
      JSON.stringify({
        etag: '"cached-etag"',
        fetchedAt: staleFetchedAt,
        catalog: TRIMMED_FIXTURE,
      }),
    );
    stubFetch(() => {
      throw new Error("network down");
    });

    const result = await fetchModelsDevCatalog();

    expect(result).toEqual(TRIMMED_FIXTURE);
  });

  it("returns null (never throws) when there is no cache and the network fails", async () => {
    stubFetch(() => {
      throw new Error("network down");
    });

    await expect(fetchModelsDevCatalog()).resolves.toBeNull();
  });

  it("returns null when the server responds with a non-OK, non-304 status and there is no cache", async () => {
    stubFetch(() => jsonResponse(undefined, { ok: false, status: 500 }));

    await expect(fetchModelsDevCatalog()).resolves.toBeNull();
  });

  it("discards a corrupted localStorage entry and fetches fresh instead of erroring", async () => {
    window.localStorage.setItem(
      MODELS_DEV_CATALOG_STORAGE_KEY,
      "{not valid json",
    );
    const fetchMock = stubFetch(() => jsonResponse(RAW_FIXTURE));

    const result = await fetchModelsDevCatalog();

    expect(result).toEqual(TRIMMED_FIXTURE);
    // No cached etag existed, so no If-None-Match should have been sent.
    expect(fetchMock).toHaveBeenCalledWith(
      "https://models.dev/api.json",
      expect.objectContaining({ headers: {} }),
    );
  });

  it("discards a localStorage entry with the wrong shape (user-edited/tampered)", async () => {
    window.localStorage.setItem(
      MODELS_DEV_CATALOG_STORAGE_KEY,
      JSON.stringify({ etag: 123, fetchedAt: "not-a-number", catalog: "nope" }),
    );
    stubFetch(() => jsonResponse(RAW_FIXTURE));

    const result = await fetchModelsDevCatalog();

    expect(result).toEqual(TRIMMED_FIXTURE);
  });

  it("degrades to an in-memory cache when localStorage.setItem throws (quota exceeded)", async () => {
    const fetchMock = stubFetch(() =>
      jsonResponse(RAW_FIXTURE, { etag: '"v1"' }),
    );

    // jsdom's Storage instance is Proxy-backed (arbitrary property writes
    // map to storage entries), so `vi.spyOn(window.localStorage, "setItem")`
    // doesn't actually override the method — stub the whole global instead
    // with a minimal store whose `setItem` always throws, simulating a full
    // quota.
    const backingStore: Record<string, string> = {};
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => backingStore[key] ?? null,
      setItem: () => {
        throw new DOMException("Quota exceeded", "QuotaExceededError");
      },
      removeItem: (key: string) => {
        delete backingStore[key];
      },
      clear: () => {
        Object.keys(backingStore).forEach((key) => delete backingStore[key]);
      },
    });

    const first = await fetchModelsDevCatalog();
    expect(first).toEqual(TRIMMED_FIXTURE);
    expect(
      window.localStorage.getItem(MODELS_DEV_CATALOG_STORAGE_KEY),
    ).toBeNull();

    // Second call within the same session should be served from the
    // in-memory fallback rather than re-fetching.
    const second = await fetchModelsDevCatalog();
    expect(second).toEqual(TRIMMED_FIXTURE);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
