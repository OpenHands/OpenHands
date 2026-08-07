import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useSearchProviders } from "#/hooks/query/use-search-providers";
import ConfigService from "#/api/config-service/config-service.api";
import type {
  LLMProvider,
  ProviderPage,
} from "#/api/config-service/config-service.types";

vi.mock("#/api/config-service/config-service.api");
vi.mock("#/hooks/query/use-verified-models", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("#/hooks/query/use-verified-models")>();
  return {
    ...actual,
    fetchVerifiedModelsByProvider: vi.fn().mockResolvedValue({}),
  };
});

const provider = (name: string): LLMProvider => ({ name, verified: false });

const page = (names: string[], nextPageId: string | null): ProviderPage => ({
  items: names.map(provider),
  next_page_id: nextPageId,
});

describe("useSearchProviders", () => {
  let queryClient: QueryClient;
  let wrapper: ({
    children,
  }: {
    children: React.ReactNode;
  }) => React.ReactElement;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    wrapper = ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it("follows next_page_id until the server stops returning one", async () => {
    const searchProviders = vi.mocked(ConfigService.searchProviders);
    searchProviders
      .mockResolvedValueOnce(page(["openhands", "anthropic"], "MTAw"))
      .mockResolvedValueOnce(page(["together_ai", "vertex_ai"], "MjAw"))
      .mockResolvedValueOnce(page(["xai"], null));

    const { result } = renderHook(() => useSearchProviders(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data?.map((p) => p.name)).toEqual([
      "openhands",
      "anthropic",
      "together_ai",
      "vertex_ai",
      "xai",
    ]);
    expect(searchProviders).toHaveBeenCalledTimes(3);
    expect(searchProviders.mock.calls[1][0]).toMatchObject({
      page_id: "MTAw",
    });
    expect(searchProviders.mock.calls[2][0]).toMatchObject({
      page_id: "MjAw",
    });
  });

  it("stops after one request when the first page has no cursor", async () => {
    const searchProviders = vi.mocked(ConfigService.searchProviders);
    searchProviders.mockResolvedValueOnce(page(["openhands"], null));

    const { result } = renderHook(() => useSearchProviders(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(searchProviders).toHaveBeenCalledTimes(1);
    expect(searchProviders).toHaveBeenNthCalledWith(
      1,
      { limit: 100, page_id: undefined },
      {},
    );
  });

  it("gives up rather than paginating forever when the cursor never clears", async () => {
    const searchProviders = vi.mocked(ConfigService.searchProviders);
    searchProviders.mockResolvedValue(page(["endless"], "always-more"));

    const { result } = renderHook(() => useSearchProviders(), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));

    expect(searchProviders).toHaveBeenCalledTimes(10);
  });
});
