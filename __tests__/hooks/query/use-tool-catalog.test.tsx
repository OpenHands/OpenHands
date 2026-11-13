import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, expect, it, vi } from "vitest";
import ToolCatalogService from "#/api/tool-catalog-service/tool-catalog-service.api";
import type { Backend } from "#/api/backend-registry/types";
import { useToolCatalog } from "#/hooks/query/use-tool-catalog";

const activeBackend = vi.hoisted(() => ({ current: null as Backend | null }));
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({ backend: activeBackend.current, orgId: null }),
}));

const local = (host: string): Backend => ({
  id: "local",
  name: "local",
  host,
  apiKey: "",
  kind: "local",
});

function wrapper({ children }: { children: ReactNode }) {
  return (
    <QueryClientProvider
      client={
        new QueryClient({ defaultOptions: { queries: { retry: false } } })
      }
    >
      {children}
    </QueryClientProvider>
  );
}

beforeEach(() => {
  vi.restoreAllMocks();
  activeBackend.current = local("http://a.test");
});

it("asks the backend for its catalog without a capability check", async () => {
  vi.spyOn(ToolCatalogService, "getCatalog").mockResolvedValue([]);
  const { result } = renderHook(() => useToolCatalog(), { wrapper });

  await waitFor(() => expect(result.current.data).toEqual([]));
  expect(ToolCatalogService.getCatalog).toHaveBeenCalledTimes(1);
});

it("refetches when the backend's host changes", async () => {
  vi.spyOn(ToolCatalogService, "getCatalog").mockResolvedValue([]);
  const { result, rerender } = renderHook(() => useToolCatalog(), {
    wrapper,
  });
  await waitFor(() => expect(result.current.data).toEqual([]));

  activeBackend.current = local("http://b.test");
  rerender();

  await waitFor(() =>
    expect(ToolCatalogService.getCatalog).toHaveBeenCalledTimes(2),
  );
});
