import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ExperimentalFeature } from "@openhands/typescript-client/clients";
import AutomationService from "#/api/automation-service/automation-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { experimentalFeaturesService } from "#/api/experimental-features-service/experimental-features-service.api";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useFactoriesSupported } from "#/hooks/query/use-factories";
import { callCloudProxy } from "#/api/cloud/proxy";

vi.mock("#/manifests/automation-interface", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("#/manifests/automation-interface")
  >()),
  hasAutomationInterface: () => true,
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

const factoriesFeature = (enabled: boolean): ExperimentalFeature => ({
  key: "factories",
  label: "Factories",
  description: "Shows Factories in the sidebar.",
  enabled,
});

function mockCapabilities(features: string[]) {
  vi.spyOn(AutomationService, "getCapabilities").mockResolvedValue({
    ready: true,
    triggerKinds: [],
    eventSources: [],
    eventTypes: [],
    triggers: {},
    features,
  });
}

function mockExperimentalFeatures(features: ExperimentalFeature[]) {
  vi.spyOn(
    experimentalFeaturesService,
    "listExperimentalFeatures",
  ).mockResolvedValue(features);
}

function mockWebClientConfig(respond: () => Promise<unknown>) {
  vi.mocked(callCloudProxy).mockImplementation(({ path }) =>
    path === WEB_CLIENT_CONFIG_PATH
      ? respond()
      : Promise.reject(new Error(`unexpected request: ${path}`)),
  );
}

function renderSupported() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return renderHook(() => useFactoriesSupported(), {
    wrapper: ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    ),
  });
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([cloudBackend, localBackend]);
  setActiveSelection({ backendId: localBackend.id });
  vi.mocked(callCloudProxy).mockReset();
});

afterEach(() => {
  vi.restoreAllMocks();
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("useFactoriesSupported on a local backend", () => {
  it("is supported when the service serves factories and the feature is on", async () => {
    mockCapabilities(["factories"]);
    mockExperimentalFeatures([factoriesFeature(true)]);

    const { result } = renderSupported();

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.isSupported).toBe(true);
  });

  it("stays off while the feature is switched off", async () => {
    mockCapabilities(["factories"]);
    mockExperimentalFeatures([factoriesFeature(false)]);

    const { result } = renderSupported();

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.isSupported).toBe(false);
  });

  it("stays off when the agent server does not list the feature", async () => {
    mockCapabilities(["factories"]);
    mockExperimentalFeatures([]);

    const { result } = renderSupported();

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.isSupported).toBe(false);
  });

  it("stays off when the service does not serve factories", async () => {
    mockCapabilities(["automationDrafts"]);
    mockExperimentalFeatures([factoriesFeature(true)]);

    const { result } = renderSupported();

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.isSupported).toBe(false);
  });

  it("stays off when the feature list cannot be read", async () => {
    mockCapabilities(["factories"]);
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockRejectedValue(new Error("boom"));

    const { result } = renderSupported();

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.isSupported).toBe(false);
  });

  it("is loading until the capabilities and the feature list have answered", async () => {
    mockCapabilities(["factories"]);
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockReturnValue(new Promise(() => {}));

    const { result } = renderSupported();

    expect(result.current.isLoading).toBe(true);
    expect(result.current.isSupported).toBe(false);
  });

  it("ignores the enterprise flag", async () => {
    mockCapabilities(["factories"]);
    mockExperimentalFeatures([factoriesFeature(false)]);
    mockWebClientConfig(async () => ({
      feature_flags: { enable_factories: true },
    }));

    const { result } = renderSupported();

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.isSupported).toBe(false);
    expect(callCloudProxy).not.toHaveBeenCalled();
  });
});

describe("useFactoriesSupported on a cloud backend", () => {
  beforeEach(() => {
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-1" });
  });

  it("is supported when the service serves factories and the instance flag is on", async () => {
    mockCapabilities(["factories"]);
    mockExperimentalFeatures([]);
    mockWebClientConfig(async () => ({
      feature_flags: { enable_factories: true },
    }));

    const { result } = renderSupported();

    await waitFor(() => expect(result.current.isSupported).toBe(true));
    expect(result.current.isLoading).toBe(false);
  });

  it.each([
    ["is false", { enable_factories: false }],
    ["is missing", {}],
  ])("stays off when the instance flag %s", async (_label, featureFlags) => {
    mockCapabilities(["factories"]);
    mockExperimentalFeatures([]);
    mockWebClientConfig(async () => ({ feature_flags: featureFlags }));

    const { result } = renderSupported();

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.isSupported).toBe(false);
  });

  it("ignores the agent server's experimental feature", async () => {
    mockCapabilities(["factories"]);
    mockExperimentalFeatures([factoriesFeature(true)]);
    mockWebClientConfig(async () => ({
      feature_flags: { enable_factories: false },
    }));

    const { result } = renderSupported();

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.isSupported).toBe(false);
  });

  it("stays off when the service does not serve factories", async () => {
    mockCapabilities(["automationDrafts"]);
    mockExperimentalFeatures([]);
    mockWebClientConfig(async () => ({
      feature_flags: { enable_factories: true },
    }));

    const { result } = renderSupported();

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.isSupported).toBe(false);
  });

  it("stays off when the instance config cannot be read", async () => {
    mockCapabilities(["factories"]);
    mockExperimentalFeatures([]);
    mockWebClientConfig(() => Promise.reject(new Error("boom")));

    const { result } = renderSupported();

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.isSupported).toBe(false);
  });

  it("is loading until the instance config has answered", () => {
    mockCapabilities(["factories"]);
    mockExperimentalFeatures([]);
    mockWebClientConfig(() => new Promise(() => {}));

    const { result } = renderSupported();

    expect(result.current.isLoading).toBe(true);
    expect(result.current.isSupported).toBe(false);
  });
});
