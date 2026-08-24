import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClientProvider, QueryClient } from "@tanstack/react-query";
import {
  useAcpCatalogModels,
  useModelsDevCatalog,
} from "#/hooks/query/use-models-dev-catalog";
import { fetchModelsDevCatalog } from "#/api/models-dev-catalog";
import type { ModelsDevCatalog } from "#/api/models-dev-catalog";
import type { ACPModelOption } from "#/constants/acp-providers";

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
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

const CATALOG: ModelsDevCatalog = {
  anthropic: {
    id: "anthropic",
    name: "Anthropic",
    models: [
      {
        id: "claude-opus-4-6",
        label: "Claude Opus 4.6",
        efforts: ["low", "high"],
      },
      { id: "claude-haiku-4-6", label: "Claude Haiku 4.6" },
    ],
  },
};

const CURATED: ACPModelOption[] = [
  { id: "claude-opus-4-6", label: "Claude Opus 4.6 (curated)" },
];

beforeEach(() => {
  vi.mocked(fetchModelsDevCatalog).mockReset();
});

describe("useModelsDevCatalog", () => {
  it("exposes the resolved catalog on success", async () => {
    vi.mocked(fetchModelsDevCatalog).mockResolvedValue(CATALOG);

    const { result } = renderHook(() => useModelsDevCatalog(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual(CATALOG);
  });

  it("resolves to data: null (not an error state) when the service can't produce a catalog", async () => {
    vi.mocked(fetchModelsDevCatalog).mockResolvedValue(null);

    const { result } = renderHook(() => useModelsDevCatalog(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toBeNull();
    expect(result.current.isError).toBe(false);
  });
});

describe("useAcpCatalogModels", () => {
  it("returns the curated list immediately while the catalog is loading", () => {
    vi.mocked(fetchModelsDevCatalog).mockReturnValue(new Promise(() => {}));

    const { result } = renderHook(
      () => useAcpCatalogModels("claude-code", CURATED),
      { wrapper: createWrapper() },
    );

    expect(result.current.catalogStatus).toBe("loading");
    expect(result.current.models).toEqual([
      {
        id: "claude-opus-4-6",
        label: "Claude Opus 4.6 (curated)",
        source: "curated",
      },
    ]);
  });

  it("merges catalog-only models in (keeping curated first) once the catalog resolves", async () => {
    vi.mocked(fetchModelsDevCatalog).mockResolvedValue(CATALOG);

    const { result } = renderHook(
      () => useAcpCatalogModels("claude-code", CURATED),
      { wrapper: createWrapper() },
    );

    await waitFor(() => expect(result.current.catalogStatus).toBe("ready"));

    expect(result.current.models).toEqual([
      {
        id: "claude-opus-4-6",
        label: "Claude Opus 4.6 (curated)",
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

  it("reports unavailable and keeps only the curated models when the catalog can't be fetched", async () => {
    vi.mocked(fetchModelsDevCatalog).mockResolvedValue(null);

    const { result } = renderHook(
      () => useAcpCatalogModels("claude-code", CURATED),
      { wrapper: createWrapper() },
    );

    await waitFor(() =>
      expect(result.current.catalogStatus).toBe("unavailable"),
    );
    expect(result.current.models).toEqual([
      {
        id: "claude-opus-4-6",
        label: "Claude Opus 4.6 (curated)",
        source: "curated",
      },
    ]);
  });

  it("stays curated-only when no acpServer is selected, even once the catalog resolves", async () => {
    vi.mocked(fetchModelsDevCatalog).mockResolvedValue(CATALOG);

    const { result } = renderHook(() => useAcpCatalogModels(null, CURATED), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.catalogStatus).toBe("ready"));
    expect(result.current.models).toEqual([
      {
        id: "claude-opus-4-6",
        label: "Claude Opus 4.6 (curated)",
        source: "curated",
      },
    ]);
  });
});
