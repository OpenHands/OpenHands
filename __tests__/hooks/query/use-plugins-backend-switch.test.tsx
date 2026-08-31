import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import PluginsManagementService from "#/api/plugins-management-service";
import PluginsService from "#/api/plugins-service";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useLocalPlugins } from "#/hooks/query/use-local-plugins";
import { usePluginFileContent } from "#/hooks/query/use-plugin-file-content";
import { usePlugins } from "#/hooks/query/use-plugins";
import { usePluginsMarketplace } from "#/hooks/query/use-plugins-marketplace";
import { PLUGINS_QUERY_KEYS } from "#/hooks/query/query-keys";

vi.mock("#/api/plugins-management-service");
vi.mock("#/api/plugins-service");

const firstBackend: Backend = {
  id: "local-one",
  name: "Local one",
  host: "http://localhost:8001",
  apiKey: "first-key",
  kind: "local",
};

const secondBackend: Backend = {
  id: "local-two",
  name: "Local two",
  host: "http://localhost:8002",
  apiKey: "second-key",
  kind: "local",
};

describe("plugin query backend isolation", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    __resetActiveStoreForTests();
    setRegisteredBackends([firstBackend, secondBackend]);
    setActiveSelection({ backendId: firstBackend.id, orgId: null });
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
  });

  afterEach(() => {
    queryClient.clear();
    vi.clearAllMocks();
    __resetActiveStoreForTests();
  });

  it("loads separate installed, local, and marketplace data after a backend switch", async () => {
    vi.mocked(PluginsManagementService.listInstalledPlugins)
      .mockResolvedValueOnce([
        {
          name: "installed-one",
          version: "1.0.0",
          description: null,
          enabled: true,
          source: "https://example.com/one.git",
          installed_at: "2026-08-24T00:00:00Z",
          install_path: "/plugins/one",
        },
      ])
      .mockResolvedValueOnce([
        {
          name: "installed-two",
          version: "1.0.0",
          description: null,
          enabled: true,
          source: "https://example.com/two.git",
          installed_at: "2026-08-24T00:00:00Z",
          install_path: "/plugins/two",
        },
      ]);
    vi.mocked(PluginsService.getLocalPlugins)
      .mockResolvedValueOnce([
        { name: "local-one", version: "1.0.0", description: "First" },
      ])
      .mockResolvedValueOnce([
        { name: "local-two", version: "1.0.0", description: "Second" },
      ]);
    vi.mocked(PluginsService.getPluginsMarketplace)
      .mockResolvedValueOnce([
        {
          name: "market-one",
          description: "First",
          source: "https://example.com/market-one.git",
          installed: false,
        },
      ])
      .mockResolvedValueOnce([
        {
          name: "market-two",
          description: "Second",
          source: "https://example.com/market-two.git",
          installed: false,
        },
      ]);
    vi.mocked(PluginsService.getPluginFileContent)
      .mockResolvedValueOnce({ kind: "text", text: "first backend" })
      .mockResolvedValueOnce({ kind: "text", text: "second backend" });

    const wrapper = ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    );
    const { result } = renderHook(
      () => ({
        installed: usePlugins(),
        local: useLocalPlugins(),
        marketplace: usePluginsMarketplace(),
        file: usePluginFileContent("/plugins/shared", "README.md"),
      }),
      { wrapper },
    );

    await waitFor(() => {
      expect(result.current.installed.data?.[0]?.name).toBe("installed-one");
      expect(result.current.local.data?.[0]?.name).toBe("local-one");
      expect(result.current.marketplace.data?.[0]?.name).toBe("market-one");
      expect(result.current.file.data?.text).toBe("first backend");
    });

    act(() => {
      setActiveSelection({ backendId: secondBackend.id, orgId: null });
    });

    await waitFor(() => {
      expect(result.current.installed.data?.[0]?.name).toBe("installed-two");
      expect(result.current.local.data?.[0]?.name).toBe("local-two");
      expect(result.current.marketplace.data?.[0]?.name).toBe("market-two");
      expect(result.current.file.data?.text).toBe("second backend");
    });

    expect(
      queryClient
        .getQueryCache()
        .findAll({ queryKey: PLUGINS_QUERY_KEYS.installed })
        .map((query) => query.queryKey),
    ).toEqual([
      [...PLUGINS_QUERY_KEYS.installed, firstBackend.id, null],
      [...PLUGINS_QUERY_KEYS.installed, secondBackend.id, null],
    ]);
    expect(
      queryClient
        .getQueryCache()
        .findAll({ queryKey: PLUGINS_QUERY_KEYS.local }),
    ).toHaveLength(2);
    expect(
      queryClient
        .getQueryCache()
        .findAll({ queryKey: PLUGINS_QUERY_KEYS.marketplace }),
    ).toHaveLength(2);
    expect(
      queryClient
        .getQueryCache()
        .findAll({ queryKey: PLUGINS_QUERY_KEYS.fileContent }),
    ).toHaveLength(2);
  });
});
