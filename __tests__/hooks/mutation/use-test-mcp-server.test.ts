import { describe, expect, it, vi, afterEach, beforeEach } from "vitest";
import React from "react";
import { renderHook, waitFor, act } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useTestMcpServer } from "#/hooks/mutation/use-test-mcp-server";
import McpService from "#/api/mcp-service/mcp-service.api";
import type { MCPServerConfig } from "#/types/mcp-server";

vi.mock("#/api/mcp-service/mcp-service.api", () => ({
  default: { testServer: vi.fn() },
}));

const SHTTP_SERVER: MCPServerConfig = {
  id: "shttp-1",
  type: "shttp",
  url: "https://mcp.example.com/mcp",
  auth: { strategy: "bearer", value: "secret-key" },
};

describe("useTestMcpServer", () => {
  let queryClient: QueryClient;
  let wrapper: ({
    children,
  }: {
    children: React.ReactNode;
  }) => React.ReactElement;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });
    wrapper = ({ children }: { children: React.ReactNode }) =>
      React.createElement(
        QueryClientProvider,
        { client: queryClient },
        children,
      );
  });

  afterEach(() => {
    queryClient.clear();
    vi.clearAllMocks();
  });

  it("returns ok=true and tool list on success", async () => {
    vi.mocked(McpService.testServer).mockResolvedValue({
      ok: true,
      tools: ["search", "fetch"],
    });

    const { result } = renderHook(() => useTestMcpServer(), { wrapper });

    act(() => {
      result.current.mutate(SHTTP_SERVER);
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(McpService.testServer).toHaveBeenCalledWith(SHTTP_SERVER);
    expect(result.current.data).toEqual({ ok: true, tools: ["search", "fetch"] });
  });

  it("returns ok=false with error_kind=timeout on timeout failure", async () => {
    vi.mocked(McpService.testServer).mockResolvedValue({
      ok: false,
      error: "timed out",
      error_kind: "timeout",
    });

    const { result } = renderHook(() => useTestMcpServer(), { wrapper });

    act(() => {
      result.current.mutate(SHTTP_SERVER);
    });

    // The mutation resolves (HTTP 200) even though ok=false — check isSuccess.
    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toEqual({
      ok: false,
      error: "timed out",
      error_kind: "timeout",
    });
  });

  it("returns ok=false with error_kind=connection on connection failure", async () => {
    vi.mocked(McpService.testServer).mockResolvedValue({
      ok: false,
      error: "ECONNREFUSED",
      error_kind: "connection",
    });

    const { result } = renderHook(() => useTestMcpServer(), { wrapper });

    act(() => {
      result.current.mutate(SHTTP_SERVER);
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toMatchObject({
      ok: false,
      error_kind: "connection",
    });
  });

  it("returns ok=false with error_kind=unknown on generic failure", async () => {
    vi.mocked(McpService.testServer).mockResolvedValue({
      ok: false,
      error: "unexpected error",
      error_kind: "unknown",
    });

    const { result } = renderHook(() => useTestMcpServer(), { wrapper });

    act(() => {
      result.current.mutate(SHTTP_SERVER);
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data).toMatchObject({
      ok: false,
      error_kind: "unknown",
    });
  });

  it("transitions to error state when the network call itself throws", async () => {
    vi.mocked(McpService.testServer).mockRejectedValue(
      new Error("Network error"),
    );

    const { result } = renderHook(() => useTestMcpServer(), { wrapper });

    await act(async () => {
      result.current.mutate(SHTTP_SERVER);
    });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.data).toBeUndefined();
  });
});
