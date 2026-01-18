// @vitest-environment happy-dom

import { describe, it, expect, afterEach, vi } from "vitest";

// Mock axios client before EventService import
vi.mock("#/api/open-hands-axios", () => ({
  openHands: { get: vi.fn() },
}));

import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { useConversationHistory } from "#/hooks/query/use-conversation-history";
import EventService from "#/api/event-service/event-service.api";
import { useUserConversation } from "#/hooks/query/use-user-conversation";


// Mock axios client BEFORE EventService import
vi.mock("#/api/open-hands-axios", () => ({
  openHands: {
    get: vi.fn(),
  },
}));

// --------------------
// Mocks
// --------------------
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
    (useUserConversation as any).mockReturnValue({
      data: { conversation_version: "V1" },
    });

    (EventService.searchEventsV1 as any).mockResolvedValue([{ id: "event-1" }]);

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
    (useUserConversation as any).mockReturnValue({
      data: { conversation_version: "V0" },
    });

    (EventService.searchEventsV0 as any).mockResolvedValue([{ id: 1 }]);

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
