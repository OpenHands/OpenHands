import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Backend } from "#/api/backend-registry/types";
import {
  __resetActiveStoreForTests,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import LLMBalanceService from "#/api/llm-balance-service";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useLLMBalance } from "#/hooks/query/use-llm-balance";

function localBackend(overrides: Partial<Backend> = {}): Backend {
  return {
    id: "local",
    name: "Local",
    host: "http://localhost:3000",
    apiKey: "initial-key",
    kind: "local",
    ...overrides,
  };
}

function makeWrapper(queryClient: QueryClient) {
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    );
  };
}

beforeEach(() => {
  window.localStorage.clear();
  __resetActiveStoreForTests();
  setRegisteredBackends([localBackend()]);
});

afterEach(() => {
  vi.restoreAllMocks();
  window.localStorage.clear();
  __resetActiveStoreForTests();
});

describe("useLLMBalance", () => {
  it("refetches a conversation balance after the active backend credentials change", async () => {
    // Arrange — balances belong to the credentials, not merely the stable
    // backend id. The same local server can report a different balance after
    // its API key rotates.
    const getBalance = vi
      .spyOn(LLMBalanceService, "getBalance")
      .mockResolvedValueOnce({
        provider: "OpenRouter",
        limit: 5,
        limitRemaining: 4,
        usage: 1,
        usageDaily: null,
        usageWeekly: null,
        usageMonthly: null,
        isFreeTier: false,
      })
      .mockResolvedValueOnce({
        provider: "OpenRouter",
        limit: 10,
        limitRemaining: 8,
        usage: 2,
        usageDaily: null,
        usageWeekly: null,
        usageMonthly: null,
        isFreeTier: false,
      });
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const { result } = renderHook(() => useLLMBalance("conversation-1"), {
      wrapper: makeWrapper(queryClient),
    });

    await waitFor(() => {
      expect(result.current.data?.limitRemaining).toBe(4);
    });

    // Act — ActiveBackendProvider increments connectionRevision when the API
    // key changes, while the backend id and conversation remain stable.
    act(() => {
      setRegisteredBackends([
        localBackend({ apiKey: "rotated-key", connectionRevision: 1 }),
      ]);
    });

    // Assert — a new cache identity must fetch the balance for the rotated
    // credentials instead of retaining the infinitely fresh old value.
    await waitFor(() => {
      expect(result.current.data?.limitRemaining).toBe(8);
    });
    expect(getBalance).toHaveBeenCalledTimes(2);
  });
});
