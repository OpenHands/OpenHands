import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useModelCatalogWarning } from "#/hooks/use-model-catalog-warning";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { callCloudProxy } from "#/api/cloud/proxy";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";

vi.mock("#/api/cloud/proxy", () => ({ callCloudProxy: vi.fn() }));

describe("useModelCatalogWarning", () => {
  let client: QueryClient;
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={client}>
      <ActiveBackendProvider>{children}</ActiveBackendProvider>
    </QueryClientProvider>
  );

  beforeEach(() => {
    vi.mocked(callCloudProxy).mockReset();
    __resetActiveStoreForTests();
    setRegisteredBackends([
      {
        id: "cloud",
        name: "Cloud",
        kind: "cloud",
        host: "https://test.example",
        apiKey: "test",
      },
      {
        id: "local",
        name: "Local",
        kind: "local",
        host: "http://localhost:8000",
        apiKey: "test",
      },
    ]);
    setActiveSelection({ backendId: "cloud", orgId: "one" });
    client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  });

  afterEach(() => {
    client.clear();
    __resetActiveStoreForTests();
  });

  it("does not infer model removal from an empty successful catalog", async () => {
    vi.mocked(callCloudProxy).mockResolvedValue({
      items: [],
      next_page_id: null,
    });
    const { result } = renderHook(useModelCatalogWarning, { wrapper });
    expect(result.current("openhands/old-model")).toBe(false);
    await waitFor(() => expect(client.isFetching()).toBe(0));
    expect(result.current("openhands/old-model")).toBe(false);
    expect(result.current("openai/custom-model")).toBe(false);
    expect(result.current(null)).toBe(false);
  });

  it("waits for all pages and matches the full route including provider slashes", async () => {
    let finish!: (value: unknown) => void;
    vi.mocked(callCloudProxy)
      .mockResolvedValueOnce({
        items: [{ provider: "openhands", name: "first" }],
        next_page_id: "next",
      })
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            finish = resolve;
          }),
      );
    const { result } = renderHook(useModelCatalogWarning, { wrapper });
    await waitFor(() => expect(callCloudProxy).toHaveBeenCalledTimes(2));
    expect(result.current("openhands/openai/gpt-4.1-mini")).toBe(false);
    await act(async () =>
      finish({
        items: [{ provider: "openhands", name: "openai/gpt-4.1-mini" }],
        next_page_id: null,
      }),
    );
    await waitFor(() => expect(result.current("openhands/removed")).toBe(true));
    expect(result.current("openhands/first")).toBe(false);
    expect(result.current("openhands/openai/gpt-4.1-mini")).toBe(false);
    expect(vi.mocked(callCloudProxy).mock.calls[1][0].path).toContain(
      "page_id=next",
    );
  });

  it("does not interpret a failed request or partial catalog as model removal", async () => {
    vi.mocked(callCloudProxy)
      .mockResolvedValueOnce({ items: [], next_page_id: "next" })
      .mockRejectedValueOnce(new Error("Service unavailable"));
    const { result } = renderHook(useModelCatalogWarning, { wrapper });
    await waitFor(() =>
      expect(
        client.getQueryState([
          "config",
          "models",
          "openhands",
          "cloud",
          0,
          "one",
        ])?.status,
      ).toBe("error"),
    );
    expect(result.current("openhands/old-model")).toBe(false);
  });

  it("does not reuse another organization's catalog while a new one is loading", async () => {
    vi.mocked(callCloudProxy).mockResolvedValueOnce({
      items: [{ provider: "openhands", name: "different-model" }],
      next_page_id: null,
    });
    const { result } = renderHook(useModelCatalogWarning, { wrapper });
    await waitFor(() => expect(result.current("openhands/model")).toBe(true));
    vi.mocked(callCloudProxy).mockImplementationOnce(
      () => new Promise(() => {}),
    );
    act(() => setActiveSelection({ backendId: "cloud", orgId: "two" }));
    expect(result.current("openhands/model")).toBe(false);
  });

  it("disables warnings and catalog reads for local backends", async () => {
    setActiveSelection({ backendId: "local", orgId: null });
    const { result } = renderHook(useModelCatalogWarning, { wrapper });
    expect(result.current("openhands/custom-route")).toBe(false);
    expect(callCloudProxy).not.toHaveBeenCalled();
  });

  it("clears a warning when a refreshed catalog lists the model", async () => {
    vi.mocked(callCloudProxy).mockResolvedValueOnce({
      items: [{ provider: "openhands", name: "different-model" }],
      next_page_id: null,
    });
    const { result } = renderHook(useModelCatalogWarning, { wrapper });
    await waitFor(() => expect(result.current("openhands/model")).toBe(true));
    vi.mocked(callCloudProxy).mockResolvedValueOnce({
      items: [{ provider: "openhands", name: "model" }],
      next_page_id: null,
    });
    await act(async () => {
      await client.invalidateQueries({ queryKey: ["config", "models"] });
    });
    await waitFor(() => expect(result.current("openhands/model")).toBe(false));
  });
});
