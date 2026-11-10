import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useConversationSystemPrompt } from "#/hooks/query/use-conversation-system-prompt";
import EventService from "#/api/event-service/event-service.api";
import type { SystemPromptEvent } from "#/types/agent-server/core/events/system-event";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";

vi.mock("#/api/event-service/event-service.api", () => ({
  default: {
    searchEvents: vi.fn(),
  },
}));

const mockSystemPromptEvent: SystemPromptEvent = {
  id: "evt-prompt",
  timestamp: "2026-01-01T00:00:00Z",
  source: "agent",
  system_prompt: {
    type: "text",
    text: "You are OpenHands agent.",
  },
  tools: [
    {
      type: "function",
      function: {
        name: "terminal",
        description: "Execute a command",
        parameters: {},
      },
    },
  ],
};

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
};

describe("useConversationSystemPrompt", () => {
  const mockConversation: AppConversation = {
    id: "conv-123",
    title: "Test Conversation",
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    conversation_url: "https://sandbox.example.com",
    session_api_key: "session-key",
  } as unknown as AppConversation;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("fetches and adapts SystemPromptEvent when found in conversation history", async () => {
    vi.mocked(EventService.searchEvents).mockResolvedValueOnce({
      items: [mockSystemPromptEvent],
      next_page_id: null,
    });

    const { result } = renderHook(
      () => useConversationSystemPrompt("conv-123", mockConversation, true),
      { wrapper: createWrapper() },
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));

    expect(EventService.searchEvents).toHaveBeenCalledWith(
      "conv-123",
      "https://sandbox.example.com",
      "session-key",
      {
        limit: 50,
        sortOrder: "TIMESTAMP",
      },
    );

    expect(result.current.data).toEqual({
      content: "You are OpenHands agent.",
      tools: [
        {
          type: "function",
          function: {
            name: "terminal",
            description: "Execute a command",
            parameters: {},
          },
        },
      ],
      openhands_version: null,
      agent_class: null,
    });
  });

  it("returns null when no SystemPromptEvent exists in the initial events", async () => {
    vi.mocked(EventService.searchEvents).mockResolvedValueOnce({
      items: [
        {
          id: "evt-user",
          timestamp: "2026-01-01T00:00:01Z",
          source: "user",
        } as unknown as SystemPromptEvent,
      ],
      next_page_id: null,
    });

    const { result } = renderHook(
      () => useConversationSystemPrompt("conv-123", mockConversation, true),
      { wrapper: createWrapper() },
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toBeNull();
  });

  it("returns null when searchEvents throws an error", async () => {
    vi.mocked(EventService.searchEvents).mockRejectedValueOnce(
      new Error("Network error"),
    );

    const { result } = renderHook(
      () => useConversationSystemPrompt("conv-123", mockConversation, true),
      { wrapper: createWrapper() },
    );

    await waitFor(() => expect(result.current.isSuccess).toBe(true));
    expect(result.current.data).toBeNull();
  });

  it("does not fetch when enabled is false", () => {
    renderHook(
      () => useConversationSystemPrompt("conv-123", mockConversation, false),
      { wrapper: createWrapper() },
    );

    expect(EventService.searchEvents).not.toHaveBeenCalled();
  });
});
