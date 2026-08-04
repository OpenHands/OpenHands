import React from "react";
import { describe, expect, it, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useFilteredEvents } from "#/hooks/use-filtered-events";
import { useEventStore } from "#/stores/use-event-store";
import type { ActionEvent, MessageEvent } from "#/types/agent-server/core";
import { SecurityRisk } from "#/types/agent-server/core";
import type { SystemPromptEvent } from "#/types/agent-server/core/events/system-event";
import { NavigationProvider } from "#/context/navigation-context";
import { seedConversationEvents } from "../helpers/seed-conversation-events";

const CONVERSATION_ID = "test-conversation-id";

function wrapper({ children }: { children: React.ReactNode }) {
  return (
    <NavigationProvider
      value={{
        currentPath: `/conversations/${CONVERSATION_ID}`,
        conversationId: CONVERSATION_ID,
        isNavigating: false,
        navigate: () => undefined,
      }}
    >
      {children}
    </NavigationProvider>
  );
}

function createUserMessage(id: string): MessageEvent {
  return {
    id,
    timestamp: `2025-07-01T00:00:${id.slice(-1).padStart(2, "0")}Z`,
    source: "user",
    llm_message: {
      role: "user",
      content: [{ type: "text", text: `User message ${id}` }],
    },
    activated_microagents: [],
    extended_content: [],
  };
}

function createAgentAction(id: string): ActionEvent {
  return {
    id,
    timestamp: "2025-07-01T00:00:02Z",
    source: "agent",
    thought: [{ type: "text", text: "Agent thought" }],
    thinking_blocks: [],
    action: {
      kind: "ExecuteBashAction",
      command: "echo test",
      is_input: false,
      timeout: null,
      reset: false,
    },
    tool_name: "execute_bash",
    tool_call_id: "call-1",
    tool_call: {
      id: "call-1",
      type: "function",
      function: { name: "execute_bash", arguments: '{"command": "echo test"}' },
    },
    llm_response_id: "response-1",
    security_risk: SecurityRisk.UNKNOWN,
  };
}

function createSystemPromptEvent(id: string): SystemPromptEvent {
  return {
    id,
    timestamp: "2025-07-01T00:00:03Z",
    source: "agent",
    system_prompt: { type: "text", text: "system prompt" },
    tools: [],
  };
}

beforeEach(() => {
  useEventStore.getState().clearEvents();
});

describe("useFilteredEvents", () => {
  describe("referential stability", () => {
    it("returns the same renderableEvents reference when uiEvents has not changed", () => {
      const event = createUserMessage("msg-1");
      seedConversationEvents("test-conversation-id", [event], [event]);

      const { result, rerender } = renderHook(() => useFilteredEvents(), { wrapper });
      const firstRenderableEvents = result.current.renderableEvents;

      rerender();

      expect(result.current.renderableEvents).toBe(firstRenderableEvents);
    });

    it("returns the same allConversationEvents reference when storeEvents has not changed", () => {
      const event = createUserMessage("msg-1");
      seedConversationEvents("test-conversation-id", [event], [event]);

      const { result, rerender } = renderHook(() => useFilteredEvents(), { wrapper });
      const firstAllConversationEvents = result.current.allConversationEvents;

      rerender();

      expect(result.current.allConversationEvents).toBe(
        firstAllConversationEvents,
      );
    });

    it("returns a new renderableEvents reference when uiEvents changes", () => {
      const firstEvent = createUserMessage("msg-1");
      seedConversationEvents("test-conversation-id", [firstEvent], [firstEvent]);

      const { result } = renderHook(() => useFilteredEvents(), { wrapper });
      const firstRenderableEvents = result.current.renderableEvents;

      const secondEvent = createAgentAction("action-2");
      act(() => {
        seedConversationEvents("test-conversation-id", [firstEvent, secondEvent], [firstEvent, secondEvent]);
      });

      expect(result.current.renderableEvents).not.toBe(firstRenderableEvents);
      expect(result.current.renderableEvents).toHaveLength(2);
    });
  });

  describe("agent-server event filtering", () => {
    it("filters renderable events from uiEvents", () => {
      const userMessage = createUserMessage("msg-1");
      const systemPrompt = createSystemPromptEvent("system-1");

      seedConversationEvents("test-conversation-id", [userMessage, systemPrompt], [userMessage, systemPrompt]);

      const { result } = renderHook(() => useFilteredEvents(), { wrapper });

      expect(result.current.renderableEvents).toEqual([userMessage]);
      expect(result.current.allConversationEvents).toEqual([
        userMessage,
        systemPrompt,
      ]);
    });

    it("uses renderable events for totalEvents", () => {
      const userMessage = createUserMessage("msg-1");
      const systemPrompt = createSystemPromptEvent("system-1");

      seedConversationEvents("test-conversation-id", [userMessage, systemPrompt], [userMessage, systemPrompt]);

      const { result } = renderHook(() => useFilteredEvents(), { wrapper });
      expect(result.current.totalEvents).toBe(1);
    });
  });

  describe("hasSubstantiveAgentActions", () => {
    it("returns false when no events exist", () => {
      const { result } = renderHook(() => useFilteredEvents(), { wrapper });
      expect(result.current.hasSubstantiveAgentActions).toBe(false);
    });

    it("returns false when only user events exist", () => {
      const userMessage = createUserMessage("msg-1");

      seedConversationEvents("test-conversation-id", [userMessage], [userMessage]);

      const { result } = renderHook(() => useFilteredEvents(), { wrapper });
      expect(result.current.hasSubstantiveAgentActions).toBe(false);
    });

    it("returns false when only system prompt events exist", () => {
      const systemPrompt = createSystemPromptEvent("system-1");

      seedConversationEvents("test-conversation-id", [systemPrompt], [systemPrompt]);

      const { result } = renderHook(() => useFilteredEvents(), { wrapper });
      expect(result.current.hasSubstantiveAgentActions).toBe(false);
    });

    it("returns true when agent action events exist", () => {
      const agentAction = createAgentAction("action-1");

      seedConversationEvents("test-conversation-id", [agentAction], [agentAction]);

      const { result } = renderHook(() => useFilteredEvents(), { wrapper });
      expect(result.current.hasSubstantiveAgentActions).toBe(true);
    });
  });

  describe("userEventsExist", () => {
    it("returns false when no events exist", () => {
      const { result } = renderHook(() => useFilteredEvents(), { wrapper });
      expect(result.current.userEventsExist).toBe(false);
    });

    it("returns true when user events exist", () => {
      const userMessage = createUserMessage("msg-1");

      seedConversationEvents("test-conversation-id", [userMessage], [userMessage]);

      const { result } = renderHook(() => useFilteredEvents(), { wrapper });
      expect(result.current.conversationUserEventsExist).toBe(true);
      expect(result.current.userEventsExist).toBe(true);
    });
  });

  describe("empty store", () => {
    it("returns empty arrays and false flags for empty store", () => {
      const { result } = renderHook(() => useFilteredEvents(), { wrapper });

      expect(result.current.renderableEvents).toEqual([]);
      expect(result.current.allConversationEvents).toEqual([]);
      expect(result.current.totalEvents).toBe(0);
      expect(result.current.hasSubstantiveAgentActions).toBe(false);
      expect(result.current.conversationUserEventsExist).toBe(false);
      expect(result.current.userEventsExist).toBe(false);
    });
  });
});
