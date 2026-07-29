import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  __resetActiveStoreForTests,
  NO_BACKEND_ID,
} from "#/api/backend-registry/active-store";
import SettingsService from "#/api/settings-service/settings-service.api";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import { useAgentSettingsSchema } from "#/hooks/query/use-agent-settings-schema";
import type { SettingsSchema } from "#/types/settings";
import { withLlmSubscriptionSchemaFields } from "#/utils/llm-subscription-schema";

const agentSchema: SettingsSchema = {
  model_name: "AgentSettings",
  sections: [
    {
      key: "llm",
      label: "LLM",
      fields: [],
    },
  ],
};

function makeWrapper(queryClient = new QueryClient()) {
  function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    );
  }
  return Wrapper;
}

function useSchemaWithBackendContext() {
  return {
    backendContext: useActiveBackendContext(),
    schemaQuery: useAgentSettingsSchema(),
  };
}

beforeEach(() => {
  window.localStorage.clear();
  vi.stubEnv("VITE_BACKEND_BASE_URL", "");
  vi.stubEnv("VITE_SESSION_API_KEY", "");
  __resetActiveStoreForTests();
});

afterEach(() => {
  window.localStorage.clear();
  vi.restoreAllMocks();
  __resetActiveStoreForTests();
});

describe("useAgentSettingsSchema", () => {
  it("waits for a configured backend, then fetches schema for that backend", async () => {
    const getSettingsSchemaSpy = vi
      .spyOn(SettingsService, "getSettingsSchema")
      .mockResolvedValue(agentSchema);

    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
      },
    });

    const { result } = renderHook(() => useSchemaWithBackendContext(), {
      wrapper: makeWrapper(queryClient),
    });

    expect(result.current.backendContext.active.backend.id).toBe(NO_BACKEND_ID);
    await waitFor(() => {
      expect(getSettingsSchemaSpy).not.toHaveBeenCalled();
    });

    act(() => {
      result.current.backendContext.addBackend({
        name: "Local",
        host: "http://localhost:8000",
        apiKey: "session-key",
        kind: "local",
      });
    });

    await waitFor(() => {
      expect(result.current.schemaQuery.data).toEqual(
        withLlmSubscriptionSchemaFields(agentSchema),
      );
    });
    expect(getSettingsSchemaSpy).toHaveBeenCalledTimes(1);
  });
});
