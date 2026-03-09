import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

vi.mock("@tanstack/react-query", async () => {
  const actual = await vi.importActual("@tanstack/react-query");
  return {
    ...actual,
    useQuery: vi.fn(),
  };
});

vi.mock("#/api/open-hands-axios", () => ({
  openHands: {
    get: vi.fn(),
  },
}));

describe("useIncidentStatus", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );

  it("calls useQuery with the correct query key", async () => {
    const { useQuery } = await import("@tanstack/react-query");
    vi.mocked(useQuery).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: null,
    } as any);

    const { useIncidentStatus } = await import(
      "#/hooks/query/use-incident-status"
    );

    renderHook(() => useIncidentStatus(), { wrapper });

    expect(useQuery).toHaveBeenCalledWith(
      expect.objectContaining({ queryKey: ["incident-status"] }),
    );
  });

  it("is enabled by default", async () => {
    const { useQuery } = await import("@tanstack/react-query");
    vi.mocked(useQuery).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: null,
    } as any);

    const { useIncidentStatus } = await import(
      "#/hooks/query/use-incident-status"
    );

    renderHook(() => useIncidentStatus(), { wrapper });

    expect(useQuery).toHaveBeenCalledWith(
      expect.objectContaining({ enabled: true }),
    );
  });

  it("can be disabled via options", async () => {
    const { useQuery } = await import("@tanstack/react-query");
    vi.mocked(useQuery).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: null,
    } as any);

    const { useIncidentStatus } = await import(
      "#/hooks/query/use-incident-status"
    );

    renderHook(() => useIncidentStatus({ enabled: false }), { wrapper });

    expect(useQuery).toHaveBeenCalledWith(
      expect.objectContaining({ enabled: false }),
    );
  });

  it("is configured with correct staleTime and refetchInterval", async () => {
    const { useQuery } = await import("@tanstack/react-query");
    vi.mocked(useQuery).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: null,
    } as any);

    const { useIncidentStatus } = await import(
      "#/hooks/query/use-incident-status"
    );

    renderHook(() => useIncidentStatus(), { wrapper });

    expect(useQuery).toHaveBeenCalledWith(
      expect.objectContaining({
        staleTime: 1000 * 60,
        refetchInterval: 1000 * 60 * 2,
      }),
    );
  });

  it("fetches from /api/v1/status using openHands axios", async () => {
    const { useQuery } = await import("@tanstack/react-query");
    const { openHands } = await import("#/api/open-hands-axios");

    const mockData = {
      ongoing_incidents: [],
      in_progress_maintenances: [],
      scheduled_maintenances: [],
    };

    vi.mocked(useQuery).mockImplementation((options: any) => {
      options.queryFn();
      return { data: mockData, isLoading: false, error: null } as any;
    });

    vi.mocked(openHands.get).mockResolvedValue({ data: mockData });

    const { useIncidentStatus } = await import(
      "#/hooks/query/use-incident-status"
    );

    renderHook(() => useIncidentStatus(), { wrapper });

    expect(openHands.get).toHaveBeenCalledWith("/api/v1/status");
  });
});
