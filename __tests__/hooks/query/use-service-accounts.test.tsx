import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import AutomationService from "#/api/automation-service/automation-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useDeploymentCapabilities } from "#/hooks/query/use-manifest-capabilities";
import { useServiceAccountOwnersEnabled } from "#/hooks/query/use-service-accounts";

const permissions = vi.hoisted(() => ({
  value: { canManage: true, userId: "user-1" as string | null },
}));

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: { getCapabilities: vi.fn() },
}));

vi.mock("#/hooks/use-automation-permissions", () => ({
  useAutomationPermissions: () => permissions.value,
}));

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-key",
  kind: "cloud",
};
const localBackend: Backend = {
  id: "local-1",
  name: "Local",
  host: "http://localhost:8000",
  apiKey: "session-key",
  kind: "local",
};

function mockFeatures(features: string[]) {
  vi.mocked(AutomationService.getCapabilities).mockResolvedValue({
    ready: true,
    triggerKinds: [],
    eventSources: [],
    eventTypes: [],
    triggers: {},
    features,
  });
}

function renderEnabled() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  // `capabilitiesLoaded` lets a test wait until the capability answer is in.
  return renderHook(
    () => ({
      enabled: useServiceAccountOwnersEnabled(),
      capabilitiesLoaded: useDeploymentCapabilities().isSuccess,
    }),
    {
      wrapper: ({ children }: { children: React.ReactNode }) => (
        <QueryClientProvider client={queryClient}>
          <ActiveBackendProvider>{children}</ActiveBackendProvider>
        </QueryClientProvider>
      ),
    },
  );
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([cloudBackend, localBackend]);
  setActiveSelection({ backendId: cloudBackend.id, orgId: "org-1" });
  permissions.value = { canManage: true, userId: "user-1" };
  mockFeatures(["serviceAccountOwners"]);
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("useServiceAccountOwnersEnabled", () => {
  it("is on for a manager of a cloud team workspace with the capability", async () => {
    const { result } = renderEnabled();

    await waitFor(() => expect(result.current.enabled).toBe(true));
  });

  it.each([
    ["without the capability", () => mockFeatures([])],
    [
      "for a member without manage_automations",
      () => {
        permissions.value = { canManage: false, userId: "user-1" };
      },
    ],
    [
      "on a personal workspace",
      () => setActiveSelection({ backendId: cloudBackend.id, orgId: "user-1" }),
    ],
    [
      "on a local backend",
      () => setActiveSelection({ backendId: localBackend.id }),
    ],
  ])("is off %s", async (_label, arrange) => {
    arrange();
    const { result } = renderEnabled();

    await waitFor(() => expect(result.current.capabilitiesLoaded).toBe(true));
    expect(result.current.enabled).toBe(false);
  });
});
