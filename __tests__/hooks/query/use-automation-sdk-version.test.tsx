import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AutomationService from "#/api/automation-service/automation-service.api";
import type { ResolvedActiveBackend } from "#/api/backend-registry/types";
import {
  AUTOMATION_SDK_VERSION_CACHE_NAMESPACE,
  useAutomationSdkVersion,
} from "#/hooks/query/use-automation-sdk-version";
import { getQueryClient, setQueryClient } from "#/query-client-config";

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: {
    getSdkVersion: vi.fn(),
  },
}));

const activeBackendMock = vi.hoisted(() => ({
  active: {
    backend: {
      id: "local-1",
      name: "Local",
      host: "http://localhost:8000",
      apiKey: "session-key",
      kind: "local",
    },
    orgId: null,
  } as ResolvedActiveBackend,
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => activeBackendMock.active,
}));

const localSdkVersionQueryKey = [
  AUTOMATION_SDK_VERSION_CACHE_NAMESPACE,
  "local-1",
  "local",
  "http://localhost:8000",
  "",
];

describe("useAutomationSdkVersion", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setQueryClient(
      new QueryClient({
        defaultOptions: { queries: { retry: false } },
      }),
    );
    activeBackendMock.active = {
      backend: {
        id: "local-1",
        name: "Local",
        host: "http://localhost:8000",
        apiKey: "session-key",
        kind: "local",
      },
      orgId: null,
    };
  });

  it("settles lookup failures as a non-fatal null result", async () => {
    vi.mocked(AutomationService.getSdkVersion).mockRejectedValue(
      new Error("automation unavailable"),
    );

    const { result } = renderHook(() => useAutomationSdkVersion());

    await waitFor(() =>
      expect(getQueryClient().getQueryState(localSdkVersionQueryKey)).toEqual(
        expect.objectContaining({ data: null, status: "success" }),
      ),
    );

    expect(result.current).toBeNull();
    expect(AutomationService.getSdkVersion).toHaveBeenCalledTimes(1);
  });

  it("does not start a query when SDK version support is unavailable", () => {
    const getSdkVersion = vi.mocked(AutomationService.getSdkVersion);
    const getSdkVersionDescriptor = Object.getOwnPropertyDescriptor(
      AutomationService,
      "getSdkVersion",
    );
    if (!getSdkVersionDescriptor) {
      throw new Error("Expected getSdkVersion to be defined");
    }

    Object.defineProperty(AutomationService, "getSdkVersion", {
      configurable: true,
      value: undefined,
    });

    try {
      const hook = renderHook(() => useAutomationSdkVersion());

      expect(hook.result.current).toBeNull();
      expect(getSdkVersion).not.toHaveBeenCalled();
      expect(getQueryClient().getQueryState(localSdkVersionQueryKey)).toEqual(
        expect.objectContaining({
          data: null,
          fetchStatus: "idle",
          status: "success",
        }),
      );

      hook.unmount();
    } finally {
      Object.defineProperty(
        AutomationService,
        "getSdkVersion",
        getSdkVersionDescriptor,
      );
    }
  });

  it("shares one request across consumers without a QueryClientProvider", async () => {
    vi.mocked(AutomationService.getSdkVersion).mockResolvedValue("1.36.3");

    const { result } = renderHook(() => ({
      first: useAutomationSdkVersion(),
      second: useAutomationSdkVersion(),
    }));

    await waitFor(() => expect(result.current.first).toBe("1.36.3"));

    expect(result.current.second).toBe("1.36.3");
    expect(AutomationService.getSdkVersion).toHaveBeenCalledTimes(1);
  });

  it("keeps the SDK version cached across hook remounts", async () => {
    vi.mocked(AutomationService.getSdkVersion).mockResolvedValue("1.36.3");

    const first = renderHook(() => useAutomationSdkVersion());
    await waitFor(() => expect(first.result.current).toBe("1.36.3"));
    first.unmount();

    const second = renderHook(() => useAutomationSdkVersion());
    expect(second.result.current).toBe("1.36.3");

    expect(AutomationService.getSdkVersion).toHaveBeenCalledTimes(1);
  });

  it("fetches a new SDK version when the active backend changes", async () => {
    vi.mocked(AutomationService.getSdkVersion)
      .mockResolvedValueOnce("1.36.3")
      .mockResolvedValueOnce("1.37.0");

    const { result, rerender } = renderHook(() => useAutomationSdkVersion());
    await waitFor(() => expect(result.current).toBe("1.36.3"));

    activeBackendMock.active = {
      backend: {
        id: "cloud-1",
        name: "Cloud",
        host: "https://app.all-hands.dev",
        apiKey: "cloud-key",
        kind: "cloud",
      },
      orgId: "org-1",
    };
    rerender();

    await waitFor(() => expect(result.current).toBe("1.37.0"));
    expect(AutomationService.getSdkVersion).toHaveBeenCalledTimes(2);
  });

  it("does not query without an available backend", () => {
    activeBackendMock.active = {
      backend: {
        id: "no-backend",
        name: "No Backend Available",
        host: "",
        apiKey: "",
        kind: "local",
      },
      orgId: null,
    };

    const { result } = renderHook(() => useAutomationSdkVersion());

    expect(result.current).toBeNull();
    expect(AutomationService.getSdkVersion).not.toHaveBeenCalled();
  });
});
