import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ConfigService from "#/api/config-service/config-service.api";
import { useSearchProviders } from "./use-search-providers";

const { fetchVerifiedModelsByProvider } = vi.hoisted(() => ({
  fetchVerifiedModelsByProvider: vi.fn(),
}));

vi.mock("#/api/config-service/config-service.api", () => ({
  default: { searchProviders: vi.fn() },
}));

vi.mock("./use-verified-models", () => ({
  VERIFIED_MODELS_GC_TIME: 60_000,
  VERIFIED_MODELS_QUERY_KEY: ["verified-models"],
  VERIFIED_MODELS_STALE_TIME: 60_000,
  fetchVerifiedModelsByProvider,
}));

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return function TestQueryClientProvider({
    children,
  }: {
    children: ReactNode;
  }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  };
}

describe("useSearchProviders", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    fetchVerifiedModelsByProvider.mockResolvedValue({});
  });

  it("adds SSYCloud as a built-in provider", async () => {
    vi.mocked(ConfigService.searchProviders).mockResolvedValue({
      items: [{ name: "openai", verified: true }],
      next_page_id: null,
    });

    const { result } = renderHook(() => useSearchProviders(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toEqual([
      { name: "openai", verified: true },
      { name: "ssycloud", verified: true },
    ]);
  });

  it("does not duplicate SSYCloud if backend metadata adds it later", async () => {
    vi.mocked(ConfigService.searchProviders).mockResolvedValue({
      items: [
        { name: "openai", verified: true },
        { name: "ssycloud", verified: false },
      ],
      next_page_id: null,
    });

    const { result } = renderHook(() => useSearchProviders(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(
      result.current.data?.filter((provider) => provider.name === "ssycloud"),
    ).toEqual([{ name: "ssycloud", verified: true }]);
  });
});
