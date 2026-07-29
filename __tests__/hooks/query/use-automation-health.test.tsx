import { describe, it, expect, vi, beforeEach } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClientProvider, QueryClient } from "@tanstack/react-query";
import { useAutomationHealth } from "#/hooks/query/use-automation-health";
import AutomationService from "#/api/automation-service/automation-service.api";

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    checkHealth: vi.fn(),
  },
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: "test-backend", kind: "local" },
    orgId: null,
  }),
}));

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

describe("useAutomationHealth", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should return healthy status when backend is available", async () => {
    vi.mocked(AutomationService.checkHealth).mockResolvedValue({ status: "ok" });

    const { result } = renderHook(() => useAutomationHealth(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data?.status).toBe("ok");
  });

  it("should return error status when backend is not available", async () => {
    vi.mocked(AutomationService.checkHealth).mockResolvedValue({
      status: "error",
      message: "Automation backend is not available",
    });

    const { result } = renderHook(() => useAutomationHealth(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(result.current.data?.status).toBe("error");
    expect(result.current.data?.message).toBe(
      "Automation backend is not available",
    );
  });
});
