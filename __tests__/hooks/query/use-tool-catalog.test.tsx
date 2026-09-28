import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ServerClient } from "@openhands/typescript-client/clients";
import ToolCatalogService from "#/api/tool-catalog-service/tool-catalog-service.api";
import type { Backend } from "#/api/backend-registry/types";
import { useToolCatalog } from "#/hooks/query/use-tool-catalog";

const activeBackend = vi.hoisted(() => ({ current: null as Backend | null }));
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({ backend: activeBackend.current, orgId: null }),
}));

const local = (id: string, host: string): Backend => ({
  id,
  name: id,
  host,
  apiKey: "",
  kind: "local",
});

const CATALOG = [
  {
    name: "terminal",
    user_selectable: true,
    usable: true,
    in_default_set: true,
  },
];

function serverInfo(capabilities: string[]) {
  return { uptime: 0, idle_time: 0, version: "1.50.0", capabilities };
}

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

describe("useToolCatalog", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    activeBackend.current = local("a", "http://a.test");
    vi.spyOn(ToolCatalogService, "getCatalog").mockResolvedValue(CATALOG);
  });

  it("asks for the catalog once server info arrives advertising it", async () => {
    let resolveInfo!: (info: ReturnType<typeof serverInfo>) => void;
    vi.spyOn(ServerClient.prototype, "getServerInfo").mockReturnValue(
      new Promise((resolve) => {
        resolveInfo = resolve;
      }) as never,
    );
    const { result } = renderHook(() => useToolCatalog(), { wrapper });
    expect(result.current.supported).toBe(false);
    expect(ToolCatalogService.getCatalog).not.toHaveBeenCalled();

    await act(async () => resolveInfo(serverInfo(["tool_catalog_v1"])));

    await waitFor(() => expect(result.current.data).toEqual(CATALOG));
    expect(result.current.supported).toBe(true);
  });

  it("re-reads the capability when the backend switches", async () => {
    vi.spyOn(ServerClient.prototype, "getServerInfo").mockImplementation(
      function getServerInfo(this: ServerClient) {
        return Promise.resolve(
          serverInfo(this.host === "http://a.test" ? ["tool_catalog_v1"] : []),
        ) as never;
      },
    );
    const { result, rerender } = renderHook(() => useToolCatalog(), {
      wrapper,
    });
    await waitFor(() => expect(result.current.supported).toBe(true));

    activeBackend.current = local("b", "http://b.test");
    rerender();

    await waitFor(() => expect(result.current.supported).toBe(false));
    expect(ToolCatalogService.getCatalog).toHaveBeenCalledTimes(1);
  });
});
