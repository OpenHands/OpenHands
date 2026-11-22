import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ReactNode } from "react";
import { useSettings } from "#/hooks/query/use-settings";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { NO_BACKEND } from "#/api/backend-registry/active-store";
import type { Settings } from "#/types/settings";

const getSettingsMock = vi.hoisted(() => vi.fn());

vi.mock("#/api/settings-service/settings-service.api", () => ({
  default: {
    getSettings: (...args: unknown[]) =>
      getSettingsMock(...args),
  },
}));

const useActiveBackendMock = vi.fn();
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => useActiveBackendMock(),
}));

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    // The hook's own `retry` predicate overrides the client-level count, but
    // `retryDelay` still applies — keep retries instant in tests.
    defaultOptions: { queries: { retry: false, retryDelay: 0 } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe("useSettings", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useActiveBackendMock.mockReturnValue({
      backend: { id: "local-1", kind: "local", host: "http://127.0.0.1:8000" },
      orgId: null,
    });
  });

  it("returns the loaded settings on success", async () => {
    getSettingsMock.mockResolvedValue({
      ...DEFAULT_SETTINGS,
      // The hook derives llm_model from agent_settings.llm.model.
      agent_settings: {
        ...(DEFAULT_SETTINGS.agent_settings ?? {}),
        llm: { model: "custom/model" },
      },
    } satisfies Settings);

    const { result } = renderHook(() => useSettings(), { wrapper });

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data?.llm_model).toBe("custom/model");
    expect(result.current.isError).toBe(false);
  });

  it("surfaces a failed load with defaults instead of pretending settings are empty", async () => {
    getSettingsMock.mockRejectedValue(new Error("backend unreachable"));

    const { result } = renderHook(() => useSettings(), { wrapper });

    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.error).toBeInstanceOf(Error);
    expect((result.current.error as Error).message).toBe(
      "backend unreachable",
    );
    expect(result.current.data).toEqual(DEFAULT_SETTINGS);

    // The exposed refetch lets the surfaces recover without a remount.
    getSettingsMock.mockResolvedValue({
      ...DEFAULT_SETTINGS,
      agent_settings: {
        ...(DEFAULT_SETTINGS.agent_settings ?? {}),
        llm: { model: "recovered/model" },
      },
    } satisfies Settings);
    await result.current.refetch();
    await waitFor(() => expect(result.current.isError).toBe(false));
    expect(result.current.data?.llm_model).toBe("recovered/model");
  });

  it("treats a 404 from a genuinely empty backend as defaults, not a failure", async () => {
    getSettingsMock.mockRejectedValue(
      Object.assign(new Error("Not found"), { status: 404 }),
    );

    const { result } = renderHook(() => useSettings(), { wrapper });

    await waitFor(() => expect(result.current.data).toEqual(DEFAULT_SETTINGS));
    expect(result.current.isError).toBe(false);
    expect(result.current.error).toBeNull();
  });

  it("reports a synthetic idle success when no backend is registered", async () => {
    useActiveBackendMock.mockReturnValue({
      backend: NO_BACKEND,
      orgId: null,
    });

    const { result } = renderHook(() => useSettings(), { wrapper });

    expect(result.current.data).toEqual(DEFAULT_SETTINGS);
    expect(result.current.isError).toBe(false);
    expect(result.current.isLoading).toBe(false);
    expect(result.current.isFetched).toBe(false);
    expect(getSettingsMock).not.toHaveBeenCalled();
  });
});
