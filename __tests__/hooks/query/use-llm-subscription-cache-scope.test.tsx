import React from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import LLMSubscriptionService from "#/api/llm-subscription-service";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useOpenAISubscriptionModels } from "#/hooks/query/use-llm-subscription-models";
import { useOpenAISubscriptionStatus } from "#/hooks/query/use-llm-subscription-status";
import { LLM_SUBSCRIPTION_QUERY_KEYS } from "#/hooks/query/query-keys";

vi.mock("#/api/llm-subscription-service", () => ({
  default: {
    getOpenAIModels: vi.fn(),
    getOpenAIStatus: vi.fn(),
  },
}));

const localBackend: Backend = {
  id: "local-1",
  name: "Local",
  host: "http://localhost:8000",
  apiKey: "local-key",
  kind: "local",
};

const cloudBackend: Backend = {
  id: "cloud-1",
  name: "Cloud",
  host: "https://app.all-hands.dev",
  apiKey: "cloud-key",
  kind: "cloud",
};

const localStatus = {
  vendor: "openai" as const,
  connected: true,
  accountEmail: "local@example.com",
  expiresAt: null,
};

const cloudStatus = {
  ...localStatus,
  accountEmail: "cloud@example.com",
};

function makeWrapper(queryClient: QueryClient) {
  return function SubscriptionQueryTestWrapper({
    children,
  }: {
    children: React.ReactNode;
  }) {
    return (
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    );
  };
}

function useSubscriptionQueries() {
  return {
    status: useOpenAISubscriptionStatus(),
    models: useOpenAISubscriptionModels(),
  };
}

describe("OpenAI subscription query cache scope", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
    setRegisteredBackends([localBackend, cloudBackend]);
    setActiveSelection({ backendId: localBackend.id, orgId: null });

    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    vi.mocked(LLMSubscriptionService.getOpenAIStatus).mockReset();
    vi.mocked(LLMSubscriptionService.getOpenAIModels).mockReset();
  });

  afterEach(() => {
    queryClient.clear();
    window.localStorage.clear();
    __resetActiveStoreForTests();
  });

  it("refetches status and models when the active backend changes", async () => {
    // Arrange
    vi.mocked(LLMSubscriptionService.getOpenAIStatus)
      .mockResolvedValueOnce(localStatus)
      .mockResolvedValueOnce(cloudStatus);
    vi.mocked(LLMSubscriptionService.getOpenAIModels)
      .mockResolvedValueOnce(["local-model"])
      .mockResolvedValueOnce(["cloud-model"]);

    const { result } = renderHook(() => useSubscriptionQueries(), {
      wrapper: makeWrapper(queryClient),
    });
    await waitFor(() => {
      expect(result.current.status.data).toEqual(localStatus);
      expect(result.current.models.data).toEqual(["local-model"]);
    });

    // Act
    act(() => {
      setActiveSelection({ backendId: cloudBackend.id, orgId: "org-a" });
    });

    // Assert
    await waitFor(() => {
      expect(result.current.status.data).toEqual(cloudStatus);
      expect(result.current.models.data).toEqual(["cloud-model"]);
    });
    expect(LLMSubscriptionService.getOpenAIStatus).toHaveBeenCalledTimes(2);
    expect(LLMSubscriptionService.getOpenAIModels).toHaveBeenCalledTimes(2);
  });

  it("refetches status and models when the active organization changes", async () => {
    // Arrange
    setActiveSelection({ backendId: cloudBackend.id, orgId: "org-a" });
    vi.mocked(LLMSubscriptionService.getOpenAIStatus)
      .mockResolvedValueOnce(cloudStatus)
      .mockResolvedValueOnce({
        ...cloudStatus,
        accountEmail: "other-org@example.com",
      });
    vi.mocked(LLMSubscriptionService.getOpenAIModels)
      .mockResolvedValueOnce(["org-a-model"])
      .mockResolvedValueOnce(["org-b-model"]);

    const { result } = renderHook(() => useSubscriptionQueries(), {
      wrapper: makeWrapper(queryClient),
    });
    await waitFor(() => {
      expect(result.current.models.data).toEqual(["org-a-model"]);
    });

    // Act
    act(() => {
      setActiveSelection({ backendId: cloudBackend.id, orgId: "org-b" });
    });

    // Assert
    await waitFor(() => {
      expect(result.current.status.data?.accountEmail).toBe(
        "other-org@example.com",
      );
      expect(result.current.models.data).toEqual(["org-b-model"]);
    });
    expect(LLMSubscriptionService.getOpenAIStatus).toHaveBeenCalledTimes(2);
    expect(LLMSubscriptionService.getOpenAIModels).toHaveBeenCalledTimes(2);
  });

  it("keeps every scoped status cache entry under the shared status prefix", async () => {
    // Arrange
    vi.mocked(LLMSubscriptionService.getOpenAIStatus)
      .mockResolvedValueOnce(localStatus)
      .mockResolvedValueOnce(cloudStatus);

    const { result } = renderHook(() => useOpenAISubscriptionStatus(), {
      wrapper: makeWrapper(queryClient),
    });
    await waitFor(() => expect(result.current.data).toEqual(localStatus));
    act(() => {
      setActiveSelection({ backendId: cloudBackend.id, orgId: "org-a" });
    });
    await waitFor(() => expect(result.current.data).toEqual(cloudStatus));

    // Act
    await queryClient.invalidateQueries({
      queryKey: LLM_SUBSCRIPTION_QUERY_KEYS.openaiStatus,
      refetchType: "none",
    });

    // Assert
    const statusQueries = queryClient.getQueryCache().findAll({
      queryKey: LLM_SUBSCRIPTION_QUERY_KEYS.openaiStatus,
    });
    expect(statusQueries).toHaveLength(2);
    expect(statusQueries.every((query) => query.state.isInvalidated)).toBe(
      true,
    );
  });
});
