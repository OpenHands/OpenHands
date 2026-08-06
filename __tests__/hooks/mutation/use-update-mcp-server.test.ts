import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import SettingsService from "#/api/settings-service/settings-service.api";
import {
  __resetMcpHealthStoreForTests,
  getMcpHealthSnapshot,
  setMcpServerHealth,
} from "#/api/mcp-health/mcp-health-store";
import { useUpdateMcpServer } from "#/hooks/mutation/use-update-mcp-server";
import type { MCPServerConfig } from "#/types/mcp-server";
import { REDACTED_MCP_SECRET_VALUE } from "#/utils/mcp-config";
import { getMcpServerHealthKey } from "#/utils/mcp-server-health-key";

const useSettingsMock = vi.fn();
vi.mock("#/hooks/query/use-settings", () => ({
  useSettings: () => useSettingsMock(),
}));

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  function Wrapper({ children }: { children: React.ReactNode }) {
    return React.createElement(
      QueryClientProvider,
      { client: queryClient },
      children,
    );
  }
  return Wrapper;
};

describe("useUpdateMcpServer", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    __resetMcpHealthStoreForTests();
    useSettingsMock.mockReturnValue({
      data: {
        mcp_config: {
          github: {
            transport: "http",
            url: "https://github.example/mcp",
            auth: {
              strategy: "bearer",
              value: REDACTED_MCP_SECRET_VALUE,
            },
          },
          docs: {
            transport: "http",
            url: "https://docs.example/mcp",
          },
        },
      },
    });
  });

  it("patches one server without reading encrypted settings or resending siblings", async () => {
    const patchSpy = vi
      .spyOn(SettingsService, "patchMcpServer")
      .mockResolvedValue(true);
    const fetchEncryptedSpy = vi.spyOn(SettingsService, "fetchSettingsFromApi");
    const { result } = renderHook(() => useUpdateMcpServer(), {
      wrapper: createWrapper(),
    });

    await result.current.mutateAsync({
      serverId: "github",
      server: {
        id: "github",
        type: "shttp",
        name: "github",
        url: "https://github.example/v2/mcp",
        auth: {
          strategy: "bearer",
          value: REDACTED_MCP_SECRET_VALUE,
        },
      },
    });

    expect(patchSpy).toHaveBeenCalledOnce();
    expect(patchSpy).toHaveBeenCalledWith("github", {
      transport: "http",
      url: "https://github.example/v2/mcp",
    });
    expect(JSON.stringify(patchSpy.mock.calls)).not.toContain(
      REDACTED_MCP_SECRET_VALUE,
    );
    expect(fetchEncryptedSpy).not.toHaveBeenCalled();
  });

  it("sends explicit auth replacement and clearing", async () => {
    const patchSpy = vi
      .spyOn(SettingsService, "patchMcpServer")
      .mockResolvedValue(true);
    const { result } = renderHook(() => useUpdateMcpServer(), {
      wrapper: createWrapper(),
    });

    await result.current.mutateAsync({
      serverId: "github",
      server: {
        id: "github",
        type: "shttp",
        name: "github",
        url: "https://github.example/mcp",
        auth: { strategy: "bearer", value: "github_pat_new" },
      },
    });
    await result.current.mutateAsync({
      serverId: "github",
      server: {
        id: "github",
        type: "shttp",
        name: "github",
        url: "https://github.example/mcp",
      },
    });

    expect(patchSpy.mock.calls[0]?.[1]).toMatchObject({
      auth: { strategy: "bearer", value: "github_pat_new" },
    });
    expect(patchSpy.mock.calls[1]?.[1]).toMatchObject({ auth: null });
  });

  it("renames with one atomic map patch", async () => {
    const configSpy = vi
      .spyOn(SettingsService, "patchMcpConfig")
      .mockResolvedValue(true);
    const serverSpy = vi.spyOn(SettingsService, "patchMcpServer");
    const { result } = renderHook(() => useUpdateMcpServer(), {
      wrapper: createWrapper(),
    });

    await result.current.mutateAsync({
      serverId: "docs",
      server: {
        id: "docs",
        type: "shttp",
        name: "reference",
        url: "https://docs.example/mcp",
      },
    });

    expect(configSpy).toHaveBeenCalledOnce();
    expect(configSpy).toHaveBeenCalledWith({
      docs: null,
      reference: {
        transport: "http",
        url: "https://docs.example/mcp",
      },
    });
    expect(serverSpy).not.toHaveBeenCalled();
  });

  it("rejects a rename collision before issuing a request", async () => {
    const configSpy = vi.spyOn(SettingsService, "patchMcpConfig");
    const serverSpy = vi.spyOn(SettingsService, "patchMcpServer");
    const { result } = renderHook(() => useUpdateMcpServer(), {
      wrapper: createWrapper(),
    });

    await expect(
      result.current.mutateAsync({
        serverId: "docs",
        server: {
          id: "docs",
          type: "shttp",
          name: "github",
          url: "https://docs.example/mcp",
        },
      }),
    ).rejects.toThrow(/already exists/i);
    expect(configSpy).not.toHaveBeenCalled();
    expect(serverSpy).not.toHaveBeenCalled();
  });

  // @spec MCP-001 — Sparse mutations preserve sibling servers
  it("allows concurrent sparse updates to different settings keys", async () => {
    const pending = new Map<string, () => void>();
    const patchSpy = vi
      .spyOn(SettingsService, "patchMcpServer")
      .mockImplementation(
        (key) =>
          new Promise<boolean>((resolve) => {
            pending.set(key, () => resolve(true));
          }),
      );
    const { result } = renderHook(() => useUpdateMcpServer(), {
      wrapper: createWrapper(),
    });

    const github = result.current.mutateAsync({
      serverId: "github",
      server: {
        id: "github",
        type: "shttp",
        name: "github",
        url: "https://github.example/v2/mcp",
      },
    });
    const docs = result.current.mutateAsync({
      serverId: "docs",
      server: {
        id: "docs",
        type: "shttp",
        name: "docs",
        url: "https://docs.example/v2/mcp",
      },
    });

    await waitFor(() => expect(patchSpy).toHaveBeenCalledTimes(2));
    pending.get("docs")?.();
    pending.get("github")?.();
    await Promise.all([github, docs]);
  });

  it("clears the server's stored health verdict after a successful save", async () => {
    vi.spyOn(SettingsService, "patchMcpServer").mockResolvedValue(true);
    const server: MCPServerConfig = {
      id: "docs",
      type: "shttp",
      name: "docs",
      url: "https://docs.example/mcp",
    };
    const key = getMcpServerHealthKey(server);
    setMcpServerHealth(key, {
      status: "healthy",
      verification: "verified",
      toolCount: 1,
      checkedAt: 1,
    });
    const { result } = renderHook(() => useUpdateMcpServer(), {
      wrapper: createWrapper(),
    });

    await result.current.mutateAsync({ serverId: server.id, server });

    await waitFor(() => expect(getMcpHealthSnapshot()[key]).toBeUndefined());
  });

  it("clears both old and new health keys after a rename", async () => {
    vi.spyOn(SettingsService, "patchMcpConfig").mockResolvedValue(true);
    const oldServer: MCPServerConfig = {
      id: "docs",
      type: "shttp",
      name: "docs",
      url: "https://docs.example/mcp",
    };
    const newServer: MCPServerConfig = { ...oldServer, name: "reference" };
    const oldKey = getMcpServerHealthKey(oldServer);
    const newKey = getMcpServerHealthKey(newServer);
    setMcpServerHealth(oldKey, {
      status: "healthy",
      verification: "verified",
      toolCount: 1,
      checkedAt: 1,
    });
    setMcpServerHealth(newKey, {
      status: "failed",
      kind: "connection",
      error: "stale",
      checkedAt: 1,
    });
    const { result } = renderHook(() => useUpdateMcpServer(), {
      wrapper: createWrapper(),
    });

    await result.current.mutateAsync({ serverId: "docs", server: newServer });

    expect(getMcpHealthSnapshot()[oldKey]).toBeUndefined();
    expect(getMcpHealthSnapshot()[newKey]).toBeUndefined();
  });
});
