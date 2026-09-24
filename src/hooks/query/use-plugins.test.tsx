import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";
import PluginsManagementService from "#/api/plugins-management-service";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { usePlugins } from "./use-plugins";
import { PLUGINS_QUERY_KEYS } from "./query-keys";

function renderUsePlugins() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>{children}</ActiveBackendProvider>
    </QueryClientProvider>
  );
  return { queryClient, ...renderHook(() => usePlugins(), { wrapper }) };
}

describe("usePlugins", () => {
  it("scopes its cache entry to the active backend", async () => {
    vi.spyOn(
      PluginsManagementService,
      "listInstalledPlugins",
    ).mockResolvedValue([]);

    const { queryClient, result } = renderUsePlugins();
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    const [key] = queryClient
      .getQueryCache()
      .getAll()
      .map((query) => query.queryKey);
    // Unscoped, a second backend reads this same entry and shows the first
    // backend's plugins.
    expect(key.slice(0, PLUGINS_QUERY_KEYS.installed.length)).toEqual([
      ...PLUGINS_QUERY_KEYS.installed,
    ]);
    expect(key.length).toBe(PLUGINS_QUERY_KEYS.installed.length + 2);
  });
});
