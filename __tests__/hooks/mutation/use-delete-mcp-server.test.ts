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
import { useDeleteMcpServer } from "#/hooks/mutation/use-delete-mcp-server";
import type { MCPServerConfig } from "#/types/mcp-server";
import { getMcpServerHealthKey } from "#/utils/mcp-server-health-key";

const syncMocks = vi.hoisted(() => ({
  updateDefaultAgentProfile: vi.fn(),
  updateEveryAgentProfile: vi.fn(),
}));
vi.mock("#/hooks/mutation/sync-agent-profiles", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("#/hooks/mutation/sync-agent-profiles")
  >()),
  ...syncMocks,
}));

/** The `mcp_server_refs` the sync gives a profile restricted to these servers. */
const syncedRefs = (refs: string[]) => {
  expect(syncMocks.updateEveryAgentProfile).toHaveBeenCalledOnce();
  const [, change] = syncMocks.updateEveryAgentProfile.mock.calls[0];
  return change({ agent_kind: "openhands", mcp_server_refs: refs });
};

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

describe("useDeleteMcpServer", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    __resetMcpHealthStoreForTests();
  });

  it("deletes by stable map key in one request", async () => {
    const deleteSpy = vi
      .spyOn(SettingsService, "deleteMcpServer")
      .mockResolvedValue(true);
    const fetchSpy = vi.spyOn(SettingsService, "fetchSettingsFromApi");
    const target: MCPServerConfig = {
      id: "github",
      type: "shttp",
      name: "github",
      url: "https://github.example/mcp",
    };
    const { result } = renderHook(() => useDeleteMcpServer(), {
      wrapper: createWrapper(),
    });

    await result.current.mutateAsync(target);

    expect(deleteSpy).toHaveBeenCalledOnce();
    expect(deleteSpy).toHaveBeenCalledWith("github");
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("drops the deleted server from every profile that lists it", async () => {
    vi.clearAllMocks();
    vi.spyOn(SettingsService, "deleteMcpServer").mockResolvedValue(true);
    const { result } = renderHook(() => useDeleteMcpServer(), {
      wrapper: createWrapper(),
    });

    await result.current.mutateAsync({
      id: "github",
      type: "shttp",
      name: "github",
      url: "https://github.example/mcp",
    });

    expect(syncedRefs(["github", "docs"])).toEqual({
      mcp_server_refs: ["docs"],
    });
  });

  it("drops the deleted server's health verdict after success", async () => {
    vi.spyOn(SettingsService, "deleteMcpServer").mockResolvedValue(true);
    const target: MCPServerConfig = {
      id: "github",
      type: "shttp",
      name: "github",
      url: "https://github.example/mcp",
    };
    const key = getMcpServerHealthKey(target);
    setMcpServerHealth(key, {
      status: "healthy",
      verification: "verified",
      toolCount: 1,
      checkedAt: 1,
    });
    const { result } = renderHook(() => useDeleteMcpServer(), {
      wrapper: createWrapper(),
    });

    await result.current.mutateAsync(target);

    await waitFor(() => expect(getMcpHealthSnapshot()[key]).toBeUndefined());
  });
});
