import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useDeleteMcpServer } from "#/hooks/mutation/use-delete-mcp-server";
import { SETTINGS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { getMcpServerHealthKey } from "#/utils/mcp-server-health-key";
import type { MCPServerConfig } from "#/types/mcp-server";

const { deleteMcpServerMock, clearMcpServerHealthMock } = vi.hoisted(() => ({
  deleteMcpServerMock: vi.fn(),
  clearMcpServerHealthMock: vi.fn(),
}));

vi.mock("#/api/settings-service/settings-service.api", () => ({
  default: {
    deleteMcpServer: (...args: unknown[]) => deleteMcpServerMock(...args),
  },
}));

vi.mock("#/api/mcp-health/mcp-health-store", () => ({
  clearMcpServerHealth: (key: string) => clearMcpServerHealthMock(key),
}));

const sseServer: MCPServerConfig = {
  id: "srv-sse",
  type: "sse",
  name: "Docs",
  url: "https://mcp.example.com/sse",
};

const stdioServer: MCPServerConfig = {
  id: "srv-stdio",
  type: "stdio",
  name: "Local",
  command: "node",
  args: ["server.js"],
};

function renderDeleteHook() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const invalidate = vi.spyOn(queryClient, "invalidateQueries");
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  const { result } = renderHook(() => useDeleteMcpServer(), { wrapper });
  return { result, invalidate };
}

beforeEach(() => {
  vi.clearAllMocks();
  deleteMcpServerMock.mockResolvedValue(undefined);
});

describe("useDeleteMcpServer", () => {
  it("deletes a server by its canonical settings-map id", async () => {
    const { result } = renderDeleteHook();

    await act(async () => {
      await result.current.mutateAsync(sseServer);
    });

    expect(deleteMcpServerMock).toHaveBeenCalledExactlyOnceWith("srv-sse");
  });

  it("clears the deleted server's health entry and refreshes personal settings on success", async () => {
    const { result, invalidate } = renderDeleteHook();

    await act(async () => {
      await result.current.mutateAsync(stdioServer);
    });

    expect(clearMcpServerHealthMock).toHaveBeenCalledExactlyOnceWith(
      getMcpServerHealthKey(stdioServer),
    );
    expect(invalidate).toHaveBeenCalledWith({
      queryKey: SETTINGS_QUERY_KEYS.personal(),
    });
  });

  it("derives a distinct health key per target so siblings are not orphaned", async () => {
    const { result } = renderDeleteHook();

    await act(async () => {
      await result.current.mutateAsync(sseServer);
    });

    const key = clearMcpServerHealthMock.mock.calls[0][0];
    expect(key).toBe(getMcpServerHealthKey(sseServer));
    expect(key).not.toBe(getMcpServerHealthKey(stdioServer));
  });

  it("does not clear health or refresh settings when the deletion request fails", async () => {
    deleteMcpServerMock.mockRejectedValueOnce(new Error("network down"));
    const { result, invalidate } = renderDeleteHook();

    await act(async () => {
      await expect(result.current.mutateAsync(sseServer)).rejects.toThrow(
        "network down",
      );
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(clearMcpServerHealthMock).not.toHaveBeenCalled();
    expect(invalidate).not.toHaveBeenCalled();
  });
});
