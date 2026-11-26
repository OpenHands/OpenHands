import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import AutomationService from "#/api/automation-service/automation-service.api";
import type { ResolvedActiveBackend } from "#/api/backend-registry/types";
import RunningVersionsService from "#/api/running-versions-service";
import { useRunningVersions } from "#/hooks/query/use-running-versions";
import { setQueryClient } from "#/query-client-config";

const { activeBackendMock } = vi.hoisted(() => ({
  activeBackendMock: { active: null as ResolvedActiveBackend | null },
}));

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: { getVersionInfo: vi.fn() },
}));
vi.mock("#/api/running-versions-service", () => ({
  default: { getBackendVersions: vi.fn() },
}));
vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => activeBackendMock.active!,
}));

const localActive: ResolvedActiveBackend = {
  backend: {
    id: "local-a",
    name: "Local A",
    host: "http://localhost:8000",
    apiKey: "key-a",
    kind: "local",
  },
  orgId: null,
};
const cloudActive: ResolvedActiveBackend = {
  backend: {
    id: "cloud-b",
    name: "Cloud B",
    host: "https://app.example.test",
    apiKey: "key-b",
    kind: "cloud",
  },
  orgId: "org-b",
};

describe("useRunningVersions", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    setQueryClient(
      new QueryClient({ defaultOptions: { queries: { retry: false } } }),
    );
    activeBackendMock.active = localActive;
    vi.mocked(RunningVersionsService.getBackendVersions).mockResolvedValue(
      null,
    );
    vi.mocked(AutomationService.getVersionInfo).mockResolvedValue(null);
  });

  it("pins each version request to the backend and organization in its query key", async () => {
    const { rerender } = renderHook(() => useRunningVersions());

    await waitFor(() => {
      expect(RunningVersionsService.getBackendVersions).toHaveBeenCalledTimes(
        1,
      );
      expect(AutomationService.getVersionInfo).toHaveBeenCalledTimes(1);
    });
    expect(RunningVersionsService.getBackendVersions).toHaveBeenNthCalledWith(
      1,
      localActive,
    );
    expect(AutomationService.getVersionInfo).toHaveBeenNthCalledWith(
      1,
      localActive,
    );

    activeBackendMock.active = cloudActive;
    rerender();

    await waitFor(() => {
      expect(RunningVersionsService.getBackendVersions).toHaveBeenCalledTimes(
        2,
      );
      expect(AutomationService.getVersionInfo).toHaveBeenCalledTimes(2);
    });
    expect(RunningVersionsService.getBackendVersions).toHaveBeenNthCalledWith(
      2,
      cloudActive,
    );
    expect(AutomationService.getVersionInfo).toHaveBeenNthCalledWith(
      2,
      cloudActive,
    );
  });
});
