import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AutomationService from "#/api/automation-service/automation-service.api";
import type { ResolvedActiveBackend } from "#/api/backend-registry/types";
import { useAutomationSdkVersion } from "#/hooks/query/use-automation-sdk-version";

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

function makeWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  }

  return { queryClient, Wrapper };
}

describe("useAutomationSdkVersion", () => {
  beforeEach(() => {
    vi.clearAllMocks();
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

  it("stores the version under the full active backend identity in React Query", async () => {
    vi.mocked(AutomationService.getSdkVersion).mockResolvedValue("1.36.3");

    const { queryClient, Wrapper } = makeWrapper();
    const { result } = renderHook(() => useAutomationSdkVersion(), {
      wrapper: Wrapper,
    });

    await waitFor(() => expect(result.current).toBe("1.36.3"));

    expect(
      queryClient.getQueryData([
        "automation-sdk-version",
        "local-1",
        "local",
        "http://localhost:8000",
        null,
      ]),
    ).toBe("1.36.3");
  });

  it("settles lookup failures as a non-fatal null result", async () => {
    vi.mocked(AutomationService.getSdkVersion).mockRejectedValue(
      new Error("automation unavailable"),
    );

    const { queryClient, Wrapper } = makeWrapper();
    const { result } = renderHook(() => useAutomationSdkVersion(), {
      wrapper: Wrapper,
    });

    await waitFor(() =>
      expect(
        queryClient.getQueryState([
          "automation-sdk-version",
          "local-1",
          "local",
          "http://localhost:8000",
          null,
        ]),
      ).toEqual(expect.objectContaining({ data: null, status: "success" })),
    );

    expect(result.current).toBeNull();
    expect(AutomationService.getSdkVersion).toHaveBeenCalledTimes(1);
  });

  it("does not start a query when SDK version support is unavailable", () => {
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
      const { queryClient, Wrapper } = makeWrapper();
      const hook = renderHook(() => useAutomationSdkVersion(), {
        wrapper: Wrapper,
      });

      expect(hook.result.current).toBeNull();
      expect(
        queryClient.getQueryState([
          "automation-sdk-version",
          "local-1",
          "local",
          "http://localhost:8000",
          null,
        ]),
      ).toEqual(
        expect.objectContaining({
          data: undefined,
          fetchStatus: "idle",
          status: "pending",
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

  it("shares one SDK version request across multiple hook consumers", async () => {
    vi.mocked(AutomationService.getSdkVersion).mockResolvedValue("1.36.3");

    const { Wrapper } = makeWrapper();
    const { result } = renderHook(
      () => ({
        first: useAutomationSdkVersion(),
        second: useAutomationSdkVersion(),
      }),
      { wrapper: Wrapper },
    );

    await waitFor(() => expect(result.current.first).toBe("1.36.3"));

    expect(result.current.second).toBe("1.36.3");
    expect(AutomationService.getSdkVersion).toHaveBeenCalledTimes(1);
  });

  it("keeps the SDK version cached across hook remounts", async () => {
    vi.mocked(AutomationService.getSdkVersion).mockResolvedValue("1.36.3");

    const { Wrapper } = makeWrapper();
    const first = renderHook(() => useAutomationSdkVersion(), {
      wrapper: Wrapper,
    });
    await waitFor(() => expect(first.result.current).toBe("1.36.3"));
    first.unmount();

    const second = renderHook(() => useAutomationSdkVersion(), {
      wrapper: Wrapper,
    });
    expect(second.result.current).toBe("1.36.3");

    expect(AutomationService.getSdkVersion).toHaveBeenCalledTimes(1);
  });

  it("fetches a new SDK version when the active backend changes", async () => {
    vi.mocked(AutomationService.getSdkVersion)
      .mockResolvedValueOnce("1.36.3")
      .mockResolvedValueOnce("1.37.0");

    const { Wrapper } = makeWrapper();
    const { result, rerender } = renderHook(() => useAutomationSdkVersion(), {
      wrapper: Wrapper,
    });
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

    const { Wrapper } = makeWrapper();
    const { result } = renderHook(() => useAutomationSdkVersion(), {
      wrapper: Wrapper,
    });

    expect(result.current).toBeNull();
    expect(AutomationService.getSdkVersion).not.toHaveBeenCalled();
  });
});
