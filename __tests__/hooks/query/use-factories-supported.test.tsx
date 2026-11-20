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

vi.mock("#/manifests/automation-interface", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("#/manifests/automation-interface")
  >()),
  hasAutomationInterface: () => true,
}));

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
  setRegisteredBackends([localBackend]);
  setActiveSelection({ backendId: localBackend.id });
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
});
