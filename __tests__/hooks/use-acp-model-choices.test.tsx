import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClientProvider, QueryClient } from "@tanstack/react-query";
import {
  buildAcpModelChoices,
  useAcpModelChoices,
} from "#/hooks/use-acp-model-choices";
import { fetchModelsDevCatalog } from "#/api/models-dev-catalog";
import type {
  MergedModelOption,
  ModelsDevCatalog,
} from "#/api/models-dev-catalog";
import type { ACPModelOption } from "#/constants/acp-providers";
import { useAcpCustomModelsStore } from "#/stores/acp-custom-models-store";

vi.mock("#/api/models-dev-catalog", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("#/api/models-dev-catalog")>();
  return {
    ...actual,
    fetchModelsDevCatalog: vi.fn(),
  };
});

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  }
  return Wrapper;
}

beforeEach(() => {
  vi.mocked(fetchModelsDevCatalog).mockReset();
  useAcpCustomModelsStore.setState({ customModelsByProfileId: {} });
});

describe("buildAcpModelChoices", () => {
  it("returns curated models tagged 'curated' when nothing else is provided", () => {
    const curated: ACPModelOption[] = [{ id: "a", label: "A" }];

    expect(buildAcpModelChoices({ curated })).toEqual([
      {
        id: "a",
        label: "A",
        source: "curated",
        description: undefined,
        efforts: undefined,
      },
    ]);
  });

  it("returns an empty list when every input is empty", () => {
    expect(buildAcpModelChoices({ curated: [] })).toEqual([]);
  });

  it("orders live -> curated -> custom -> models.dev extras", () => {
    const catalogExtras: MergedModelOption[] = [
      { id: "extra-1", label: "Extra One", source: "models.dev" },
    ];

    const result = buildAcpModelChoices({
      liveModels: [{ id: "live-1", label: "Live One" }],
      curated: [{ id: "curated-1", label: "Curated One" }],
      customIds: ["custom-1"],
      catalogExtras,
    });

    expect(result.map((choice) => choice.id)).toEqual([
      "live-1",
      "curated-1",
      "custom-1",
      "extra-1",
    ]);
    expect(result.map((choice) => choice.source)).toEqual([
      "live",
      "curated",
      "custom",
      "models.dev",
    ]);
  });

  it("dedupes a shared id with precedence live > curated > custom > catalog", () => {
    const catalogExtras: MergedModelOption[] = [
      { id: "shared", label: "Catalog label", source: "models.dev" },
    ];

    const result = buildAcpModelChoices({
      liveModels: [{ id: "shared", label: "Live label" }],
      curated: [{ id: "shared", label: "Curated label" }],
      customIds: ["shared"],
      catalogExtras,
    });

    expect(result).toEqual([
      {
        id: "shared",
        label: "Live label",
        source: "live",
        description: undefined,
        efforts: undefined,
      },
    ]);
  });

  it("falls through precedence tiers when higher-precedence sources are absent", () => {
    const catalogExtras: MergedModelOption[] = [
      { id: "shared", label: "Catalog label", source: "models.dev" },
    ];

    const result = buildAcpModelChoices({
      curated: [{ id: "shared", label: "Curated label" }],
      customIds: ["shared"],
      catalogExtras,
    });

    expect(result).toEqual([
      {
        id: "shared",
        label: "Curated label",
        source: "curated",
        description: undefined,
        efforts: undefined,
      },
    ]);
  });

  it("dedupes a custom id against a catalog extra when there's no curated match", () => {
    const catalogExtras: MergedModelOption[] = [
      { id: "shared", label: "Catalog label", source: "models.dev" },
    ];

    const result = buildAcpModelChoices({
      curated: [],
      customIds: ["shared"],
      catalogExtras,
    });

    expect(result).toEqual([
      {
        id: "shared",
        label: "shared",
        source: "custom",
        description: undefined,
        efforts: undefined,
      },
    ]);
  });

  it("drops a catalog extra whose label matches a curated entry (alias vs full id)", () => {
    const catalogExtras: MergedModelOption[] = [
      {
        id: "claude-sonnet-4-6",
        label: "Claude Sonnet 4.6",
        source: "models.dev",
      },
      { id: "claude-other-model", label: "Claude Other", source: "models.dev" },
    ];

    const result = buildAcpModelChoices({
      curated: [{ id: "sonnet", label: "Claude Sonnet 4.6" }],
      catalogExtras,
    });

    expect(result.map((c) => c.id)).toEqual(["sonnet", "claude-other-model"]);
  });

  it("label-dedupes catalog extras case-insensitively but never drops live/curated/custom entries", () => {
    const result = buildAcpModelChoices({
      liveModels: [{ id: "live-1", label: "Shared Label" }],
      curated: [{ id: "curated-1", label: "shared label" }],
      customIds: ["custom-1"],
      catalogExtras: [
        { id: "cat-1", label: "SHARED LABEL", source: "models.dev" },
        { id: "cat-2", label: "custom-1 ", source: "models.dev" },
      ],
    });

    expect(result.map((c) => c.id)).toEqual([
      "live-1",
      "curated-1",
      "custom-1",
    ]);
  });

  it("labels a custom entry with its id (no separate display name is stored)", () => {
    const result = buildAcpModelChoices({
      curated: [],
      customIds: ["my-custom-model"],
    });

    expect(result).toEqual([
      {
        id: "my-custom-model",
        label: "my-custom-model",
        source: "custom",
        description: undefined,
        efforts: undefined,
      },
    ]);
  });

  it("carries description/efforts metadata through from catalog extras", () => {
    const catalogExtras: MergedModelOption[] = [
      {
        id: "extra-1",
        label: "Extra One",
        source: "models.dev",
        description: "desc",
        efforts: ["low", "high"],
      },
    ];

    const result = buildAcpModelChoices({ curated: [], catalogExtras });

    expect(result).toEqual([
      {
        id: "extra-1",
        label: "Extra One",
        source: "models.dev",
        description: "desc",
        efforts: ["low", "high"],
      },
    ]);
  });

  it("ignores blank/whitespace-only ids from every source", () => {
    const result = buildAcpModelChoices({
      curated: [{ id: "   ", label: "Blank" }],
      customIds: [""],
    });

    expect(result).toEqual([]);
  });
});

describe("useAcpModelChoices", () => {
  const CURATED: ACPModelOption[] = [{ id: "curated-1", label: "Curated One" }];

  it("returns curated-only choices while the catalog is loading", () => {
    vi.mocked(fetchModelsDevCatalog).mockReturnValue(new Promise(() => {}));

    const { result } = renderHook(
      () => useAcpModelChoices({ acpServer: "claude-code", curated: CURATED }),
      { wrapper: createWrapper() },
    );

    expect(result.current.catalogStatus).toBe("loading");
    expect(result.current.choices).toEqual([
      {
        id: "curated-1",
        label: "Curated One",
        source: "curated",
        description: undefined,
        efforts: undefined,
      },
    ]);
  });

  it("upgrades in place with models.dev extras once the catalog resolves", async () => {
    const catalog: ModelsDevCatalog = {
      anthropic: {
        id: "anthropic",
        name: "Anthropic",
        models: [
          { id: "curated-1", label: "Curated One (catalog)" },
          { id: "extra-1", label: "Extra One" },
        ],
      },
    };
    vi.mocked(fetchModelsDevCatalog).mockResolvedValue(catalog);

    const { result } = renderHook(
      () => useAcpModelChoices({ acpServer: "claude-code", curated: CURATED }),
      { wrapper: createWrapper() },
    );

    await waitFor(() => expect(result.current.catalogStatus).toBe("ready"));

    // The curated entry's own label wins over the catalog's duplicate id.
    expect(result.current.choices).toEqual([
      {
        id: "curated-1",
        label: "Curated One",
        source: "curated",
        description: undefined,
        efforts: undefined,
      },
      {
        id: "extra-1",
        label: "Extra One",
        source: "models.dev",
        description: undefined,
        efforts: undefined,
      },
    ]);
  });

  it("includes remembered custom models for the given profile id", async () => {
    vi.mocked(fetchModelsDevCatalog).mockResolvedValue(null);
    useAcpCustomModelsStore
      .getState()
      .addCustomModel("profile-1", "custom-model-a");

    const { result } = renderHook(
      () =>
        useAcpModelChoices({
          acpServer: "claude-code",
          curated: CURATED,
          profileId: "profile-1",
        }),
      { wrapper: createWrapper() },
    );

    await waitFor(() =>
      expect(result.current.catalogStatus).toBe("unavailable"),
    );

    expect(result.current.choices).toEqual([
      {
        id: "curated-1",
        label: "Curated One",
        source: "curated",
        description: undefined,
        efforts: undefined,
      },
      {
        id: "custom-model-a",
        label: "custom-model-a",
        source: "custom",
        description: undefined,
        efforts: undefined,
      },
    ]);
  });

  it("does not offer custom models when no profileId is given", async () => {
    vi.mocked(fetchModelsDevCatalog).mockResolvedValue(null);
    useAcpCustomModelsStore
      .getState()
      .addCustomModel("profile-1", "custom-model-a");

    const { result } = renderHook(
      () => useAcpModelChoices({ acpServer: "claude-code", curated: CURATED }),
      { wrapper: createWrapper() },
    );

    await waitFor(() =>
      expect(result.current.catalogStatus).toBe("unavailable"),
    );

    expect(result.current.choices.map((choice) => choice.id)).toEqual([
      "curated-1",
    ]);
  });

  it("passes liveModels through as the highest-precedence source", async () => {
    vi.mocked(fetchModelsDevCatalog).mockResolvedValue(null);

    const { result } = renderHook(
      () =>
        useAcpModelChoices({
          acpServer: "claude-code",
          curated: CURATED,
          liveModels: [{ id: "live-1", label: "Live One" }],
        }),
      { wrapper: createWrapper() },
    );

    await waitFor(() =>
      expect(result.current.catalogStatus).toBe("unavailable"),
    );

    expect(result.current.choices.map((choice) => choice.source)).toEqual([
      "live",
      "curated",
    ]);
  });
});
