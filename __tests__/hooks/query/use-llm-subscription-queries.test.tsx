import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import LLMSubscriptionService, {
  type LLMSubscriptionStatus,
} from "#/api/llm-subscription-service";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { LLM_SUBSCRIPTION_QUERY_KEYS } from "#/hooks/query/query-keys";
import { useOpenAISubscriptionModels } from "#/hooks/query/use-llm-subscription-models";
import { useOpenAISubscriptionStatus } from "#/hooks/query/use-llm-subscription-status";

vi.mock("#/api/llm-subscription-service");

const localBackend1: Backend = {
  id: "local-1",
  name: "Local 1",
  host: "http://localhost:8000",
  apiKey: "session-key-1",
  kind: "local",
};

const localBackend2: Backend = {
  id: "local-2",
  name: "Local 2",
  host: "http://localhost:9000",
  apiKey: "session-key-2",
  kind: "local",
};

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "Cloud",
  host: "https://app.all-hands.dev",
  apiKey: "cloud-key",
  kind: "cloud",
};

const connectedStatus = (accountEmail: string): LLMSubscriptionStatus => ({
  vendor: "openai",
  connected: true,
  accountEmail,
  expiresAt: null,
});

describe("OpenAI subscription queries", () => {
  let queryClient: QueryClient;
  let wrapper: ({
    children,
  }: {
    children: React.ReactNode;
  }) => React.ReactElement;

  beforeEach(() => {
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend1, localBackend2, cloudBackend]);
    setActiveSelection({ backendId: localBackend1.id, orgId: null });

    queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
      },
    });
    wrapper = ({ children }: { children: React.ReactNode }) =>
      React.createElement(
        QueryClientProvider,
        { client: queryClient },
        React.createElement(ActiveBackendProvider, null, children),
      );
  });

  afterEach(() => {
    queryClient.clear();
    vi.resetAllMocks();
    __resetActiveStoreForTests();
  });

  it("refetches subscription status into a backend-scoped cache", async () => {
    vi.mocked(LLMSubscriptionService.getOpenAIStatus)
      .mockResolvedValueOnce(connectedStatus("local-1@example.com"))
      .mockResolvedValueOnce(connectedStatus("local-2@example.com"));

    const { result, rerender } = renderHook(
      () => useOpenAISubscriptionStatus(),
      { wrapper },
    );

    await waitFor(() => {
      expect(result.current.data?.accountEmail).toBe("local-1@example.com");
    });

    act(() => {
      setActiveSelection({ backendId: localBackend2.id, orgId: null });
    });
    rerender();

    await waitFor(() => {
      expect(result.current.data?.accountEmail).toBe("local-2@example.com");
    });
    expect(LLMSubscriptionService.getOpenAIStatus).toHaveBeenCalledTimes(2);
    expect(
      queryClient
        .getQueryCache()
        .findAll({ queryKey: LLM_SUBSCRIPTION_QUERY_KEYS.openaiStatus })
        .map((query) => query.queryKey),
    ).toEqual([
      [...LLM_SUBSCRIPTION_QUERY_KEYS.openaiStatus, localBackend1.id, null],
      [...LLM_SUBSCRIPTION_QUERY_KEYS.openaiStatus, localBackend2.id, null],
    ]);
  });

  it("refetches subscription models into an organization-scoped cache", async () => {
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-1" });
    vi.mocked(LLMSubscriptionService.getOpenAIModels)
      .mockResolvedValueOnce(["openai/model-for-org-1"])
      .mockResolvedValueOnce(["openai/model-for-org-2"]);

    const { result, rerender } = renderHook(
      () => useOpenAISubscriptionModels(),
      { wrapper },
    );

    await waitFor(() => {
      expect(result.current.data).toEqual(["openai/model-for-org-1"]);
    });

    act(() => {
      setActiveSelection({ backendId: cloudBackend.id, orgId: "org-2" });
    });
    rerender();

    await waitFor(() => {
      expect(result.current.data).toEqual(["openai/model-for-org-2"]);
    });
    expect(LLMSubscriptionService.getOpenAIModels).toHaveBeenCalledTimes(2);
    expect(
      queryClient
        .getQueryCache()
        .findAll({ queryKey: LLM_SUBSCRIPTION_QUERY_KEYS.openaiModels })
        .map((query) => query.queryKey),
    ).toEqual([
      [...LLM_SUBSCRIPTION_QUERY_KEYS.openaiModels, cloudBackend.id, "org-1"],
      [...LLM_SUBSCRIPTION_QUERY_KEYS.openaiModels, cloudBackend.id, "org-2"],
    ]);
  });
});
