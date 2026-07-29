import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AutomationService from "#/api/automation-service/automation-service.api";
import type { ResolvedActiveBackend } from "#/api/backend-registry/types";
import {
  __resetAutomationSdkVersionCacheForTests,
  useAutomationSdkVersion,
} from "#/hooks/query/use-automation-sdk-version";

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

describe("useAutomationSdkVersion", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    __resetAutomationSdkVersionCacheForTests();
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

  it("shares one SDK version request across multiple hook consumers", async () => {
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
