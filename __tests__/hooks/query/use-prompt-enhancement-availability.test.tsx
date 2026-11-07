import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import PromptEnhancementService from "#/api/prompt-enhancement-service";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import { usePromptEnhancementAvailability } from "#/hooks/query/use-prompt-enhancement-availability";

const acpContext = vi.hoisted(() => ({ isAcpContext: false }));

vi.mock("#/hooks/use-acp-model-context", () => ({
  useAcpModelContext: () => acpContext,
}));

vi.mock("#/hooks/use-chat-input-llm-profile-state", () => ({
  useChatInputLlmProfileState: () => ({ currentProfileName: "default" }),
}));

function useBackend(kind: "local" | "cloud") {
  setRegisteredBackends([
    {
      id: `${kind}-1`,
      name: kind,
      host: kind === "local" ? "http://127.0.0.1:18000" : "https://app.example",
      apiKey: "key",
      kind,
    },
  ]);
  setActiveSelection({ backendId: `${kind}-1`, orgId: null });
}

function renderAvailability() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return renderHook(() => usePromptEnhancementAvailability(), {
    wrapper: ({ children }: { children: React.ReactNode }) => (
      <QueryClientProvider client={client}>
        <ActiveBackendProvider>{children}</ActiveBackendProvider>
      </QueryClientProvider>
    ),
  });
}

describe("usePromptEnhancementAvailability", () => {
  beforeEach(() => {
    window.localStorage.clear();
    __resetActiveStoreForTests();
    acpContext.isAcpContext = false;
  });

  afterEach(() => {
    vi.restoreAllMocks();
    __resetActiveStoreForTests();
  });

  it("is available when the local Agent Server can use the active profile", async () => {
    // Arrange
    useBackend("local");
    const checkSpy = vi
      .spyOn(PromptEnhancementService, "checkAvailability")
      .mockResolvedValue({ available: true });

    // Act
    const { result } = renderAvailability();

    // Assert
    await waitFor(() =>
      expect(result.current).toEqual({
        isAvailable: true,
        profileName: "default",
      }),
    );
    expect(checkSpy).toHaveBeenCalledWith("default");
  });

  it("reports an Agent Server without the capability as unsupported", async () => {
    // Arrange
    useBackend("local");
    vi.spyOn(PromptEnhancementService, "checkAvailability").mockResolvedValue({
      available: false,
      code: "unsupported_backend",
      message: "This Agent Server does not support prompt enhancement.",
    });

    // Act
    const { result } = renderAvailability();

    // Assert
    await waitFor(() =>
      expect(result.current).toEqual({
        isAvailable: false,
        reason: "unsupported_backend",
      }),
    );
  });

  it.each([
    { kind: "cloud" as const, isAcp: false, reason: "cloud_backend" },
    { kind: "local" as const, isAcp: true, reason: "acp_agent" },
  ])(
    "is unavailable without a server check for $reason",
    ({ kind, isAcp, reason }) => {
      // Arrange
      useBackend(kind);
      acpContext.isAcpContext = isAcp;
      const checkSpy = vi.spyOn(PromptEnhancementService, "checkAvailability");

      // Act
      const { result } = renderAvailability();

      // Assert
      expect(result.current).toEqual({ isAvailable: false, reason });
      expect(checkSpy).not.toHaveBeenCalled();
    },
  );
});
