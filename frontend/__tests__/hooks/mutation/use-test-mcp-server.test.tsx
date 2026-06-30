import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi, beforeEach } from "vitest";
import McpTestService from "#/api/mcp-test-service/mcp-test-service.api";
import { useTestMcpServer } from "#/hooks/mutation/use-test-mcp-server";

describe("useTestMcpServer", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  const createWrapper = (queryClient: QueryClient) =>
    function Wrapper({ children }: { children: React.ReactNode }) {
      return (
        <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
      );
    };

  it("marks server health as testing in the query cache when a test starts", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false },
      },
    });

    queryClient.setQueryData(["mcp-server-health", "demo"], {
      server_id: "demo",
      status: "healthy",
      test_id: "old-test",
    });

    vi.spyOn(McpTestService, "startTest").mockResolvedValue({
      test_id: "new-test",
      status: "running",
    });

    const { result } = renderHook(() => useTestMcpServer(), {
      wrapper: createWrapper(queryClient),
    });

    result.current.mutate("demo");

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(queryClient.getQueryData(["mcp-server-health", "demo"])).toEqual({
      server_id: "demo",
      status: "testing",
      test_id: "new-test",
      tested_at: expect.any(String),
    });
  });
});
