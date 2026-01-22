import { describe, it, expect, afterEach, vi } from "vitest";
import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { useConversationHistory } from "#/hooks/query/use-conversation-history";
import EventService from "#/api/event-service/event-service.api";
import { useUserConversation } from "#/hooks/query/use-user-conversation";
import type { Conversation } from "#/api/open-hands.types";
import type { OpenHandsEvent } from "#/types/v1/core";

function makeConversation(version: "V0" | "V1"): Conversation {
  return {
    conversation_id: "conv-test",
    title: "Test Conversation",
    selected_repository: null,
    selected_branch: null,
    git_provider: null,
    last_updated_at: new Date().toISOString(),
    created_at: new Date().toISOString(),
    status: "RUNNING",
    runtime_status: null,
    url: null,
    session_api_key: null,
    conversation_version: version,
  };
}

function makeEvent(): OpenHandsEvent {
  return {
    id: "evt-1",
  } as OpenHandsEvent;
}

// --------------------
// Mocks
// --------------------
vi.mock("#/api/open-hands-axios", () => ({
  openHands: {
    get: vi.fn(),
  },
}));

vi.mock("#/api/event-service/event-service.api");
vi.mock("#/hooks/query/use-user-conversation");

const queryClient = new QueryClient();

function wrapper({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

// --------------------
// Tests
// --------------------
describe("useConversationHistory", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("calls V1 REST endpoint for V1 conversations", async () => {
    const v1SearchEventsSpy = vi.spyOn(EventService, "searchEventsV1");

    vi.mocked(useUserConversation).mockReturnValue({
      data: makeConversation("V1"),
      isLoading: false,
      isPending: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as any);

    v1SearchEventsSpy.mockResolvedValue([makeEvent()]);

    const { result } = renderHook(() => useConversationHistory("conv-123"), {
      wrapper,
    });

    await waitFor(() => {
      expect(result.current.data).toBeDefined();
    });

    expect(EventService.searchEventsV1).toHaveBeenCalledWith("conv-123");
    expect(EventService.searchEventsV0).not.toHaveBeenCalled();
  });

  it("calls V0 REST endpoint for V0 conversations", async () => {
    const v0SearchEventsSpy = vi.spyOn(EventService, "searchEventsV0");

    vi.mocked(useUserConversation).mockReturnValue({
      data: makeConversation("V0"),
      isLoading: false,
      isPending: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as any);

    v0SearchEventsSpy.mockResolvedValue([makeEvent()]);

    const { result } = renderHook(() => useConversationHistory("conv-456"), {
      wrapper,
    });

    await waitFor(() => {
      expect(result.current.data).toBeDefined();
    });

    expect(EventService.searchEventsV0).toHaveBeenCalledWith("conv-456");
    expect(EventService.searchEventsV1).not.toHaveBeenCalled();
  });
});
