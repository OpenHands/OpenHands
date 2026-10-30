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
import { useServiceAccountRunAsEnabled } from "#/hooks/query/use-service-accounts";
import { CLOUD_INTEGRATIONS_CONFIG_QUERY_KEY } from "#/hooks/query/use-cloud-integrations-config";
import { callCloudProxy } from "#/api/cloud/proxy";

const permissions = vi.hoisted(() => ({
  value: { canManage: true, userId: "user-1" as string | null },
}));

vi.mock("#/api/automation-service/automation-service.api", () => ({
  default: { getCapabilities: vi.fn() },
}));

vi.mock("#/hooks/use-automation-permissions", () => ({
  useAutomationPermissions: () => permissions.value,
}));

vi.mock("#/api/cloud/proxy", () => ({ callCloudProxy: vi.fn() }));

const WEB_CLIENT_CONFIG_PATH = "/api/v1/web-client/config";

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

function mockWebClientConfig(respond: () => Promise<unknown>) {
  vi.mocked(callCloudProxy).mockImplementation(({ path }) =>
    path === WEB_CLIENT_CONFIG_PATH
      ? respond()
      : Promise.reject(new Error(`unexpected request: ${path}`)),
  );
}

function mockServiceAccountsFlag(featureFlags: Record<string, unknown>) {
  mockWebClientConfig(async () => ({ feature_flags: featureFlags }));
}

function webClientConfigRequests() {
  return vi
    .mocked(callCloudProxy)
    .mock.calls.filter(([req]) => req.path === WEB_CLIENT_CONFIG_PATH);
}

function renderEnabled() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  // `capabilitiesLoaded` lets a test wait until the capability answer is in.
  const rendered = renderHook(
    () => ({
      enabled: useServiceAccountRunAsEnabled(),
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
  /** Resolves once the web-client config request has succeeded or failed. */
  const configSettled = () =>
    waitFor(() => {
      expect(webClientConfigRequests()).toHaveLength(1);
      expect(
        queryClient.getQueryCache().find({
          queryKey: [...CLOUD_INTEGRATIONS_CONFIG_QUERY_KEY],
        })?.state.status,
      ).not.toBe("pending");
    });
  return { ...rendered, configSettled };
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([cloudBackend, localBackend]);
  setActiveSelection({ backendId: cloudBackend.id, orgId: "org-1" });
  permissions.value = { canManage: true, userId: "user-1" };
  mockFeatures(["serviceAccountRunAs"]);
  vi.mocked(callCloudProxy).mockReset();
  mockServiceAccountsFlag({ enable_service_accounts: true });
});

afterEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("useServiceAccountRunAsEnabled", () => {
  it.each([
    [
      "a local backend",
      () => setActiveSelection({ backendId: localBackend.id }),
    ],
    [
      "a personal workspace",
      () => setActiveSelection({ backendId: cloudBackend.id, orgId: "user-1" }),
    ],
  ])(
    "requests neither capabilities nor config on %s",
    async (_label, arrange) => {
      arrange();
      const queryClient = new QueryClient();
      const { result } = renderHook(() => useServiceAccountRunAsEnabled(), {
        wrapper: ({ children }: { children: React.ReactNode }) => (
          <QueryClientProvider client={queryClient}>
            <ActiveBackendProvider>{children}</ActiveBackendProvider>
          </QueryClientProvider>
        ),
      });

      await waitFor(() => expect(result.current).toBe(false));
      expect(AutomationService.getCapabilities).not.toHaveBeenCalled();
      expect(webClientConfigRequests()).toHaveLength(0);
    },
  );

  it("is on for a manager of a cloud team workspace with the capability and the flag", async () => {
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

  it.each([
    [
      "the flag is false",
      () => mockServiceAccountsFlag({ enable_service_accounts: false }),
    ],
    ["the flag is missing", () => mockServiceAccountsFlag({})],
    [
      "the config has no feature flags",
      () => mockWebClientConfig(async () => ({})),
    ],
    [
      "the config request fails",
      () =>
        mockWebClientConfig(() =>
          Promise.reject(new Error("config unavailable")),
        ),
    ],
  ])("is off when %s", async (_label, arrange) => {
    arrange();
    const { result, configSettled } = renderEnabled();

    await waitFor(() => expect(result.current.capabilitiesLoaded).toBe(true));
    await configSettled();
    expect(result.current.enabled).toBe(false);
  });

  it("is off while the config request is loading", async () => {
    mockWebClientConfig(() => new Promise(() => {}));
    const { result } = renderEnabled();

    await waitFor(() => expect(result.current.capabilitiesLoaded).toBe(true));
    expect(webClientConfigRequests()).toHaveLength(1);
    expect(result.current.enabled).toBe(false);
  });
});
