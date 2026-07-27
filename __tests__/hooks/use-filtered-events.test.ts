import { describe, expect, it, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useFilteredEvents } from "#/hooks/use-filtered-events";
import { useEventStore } from "#/stores/use-event-store";
import type { ActionEvent, MessageEvent } from "#/types/agent-server/core";
import { SecurityRisk } from "#/types/agent-server/core";
import type { SystemPromptEvent } from "#/types/agent-server/core/events/system-event";

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
  useEventStore.setState({
    events: [],
    eventIds: new Set(),
    uiEvents: [],
  });
});

describe("useFilteredEvents", () => {
  describe("referential stability", () => {
    it("returns the same renderableEvents reference when uiEvents has not changed", () => {
      const event = createUserMessage("msg-1");
      useEventStore.setState({
        events: [event],
        eventIds: new Set(["msg-1"]),
        uiEvents: [event],
      });

      const { result, rerender } = renderHook(() => useFilteredEvents());
      const firstRenderableEvents = result.current.renderableEvents;

      rerender();

      expect(result.current.renderableEvents).toBe(firstRenderableEvents);
    });

    it("returns the same allConversationEvents reference when storeEvents has not changed", () => {
      const event = createUserMessage("msg-1");
      useEventStore.setState({
        events: [event],
        eventIds: new Set(["msg-1"]),
        uiEvents: [event],
      });

      const { result, rerender } = renderHook(() => useFilteredEvents());
      const firstAllConversationEvents = result.current.allConversationEvents;

      rerender();

      expect(result.current.allConversationEvents).toBe(
        firstAllConversationEvents,
      );
    });

    it("returns a new renderableEvents reference when uiEvents changes", () => {
      const firstEvent = createUserMessage("msg-1");
      useEventStore.setState({
        events: [firstEvent],
        eventIds: new Set(["msg-1"]),
        uiEvents: [firstEvent],
      });

      const { result } = renderHook(() => useFilteredEvents());
      const firstRenderableEvents = result.current.renderableEvents;

      const secondEvent = createAgentAction("action-2");
      act(() => {
        useEventStore.setState({
          events: [firstEvent, secondEvent],
          eventIds: new Set(["msg-1", "action-2"]),
          uiEvents: [firstEvent, secondEvent],
        });
      });

      expect(result.current.renderableEvents).not.toBe(firstRenderableEvents);
      expect(result.current.renderableEvents).toHaveLength(2);
    });
  });

  describe("agent-server event filtering", () => {
    it("filters renderable events from uiEvents", () => {
      const userMessage = createUserMessage("msg-1");
      const systemPrompt = createSystemPromptEvent("system-1");

      useEventStore.setState({
        events: [userMessage, systemPrompt],
        eventIds: new Set(["msg-1", "system-1"]),
        uiEvents: [userMessage, systemPrompt],
      });

      const { result } = renderHook(() => useFilteredEvents());

      expect(result.current.renderableEvents).toEqual([userMessage]);
      expect(result.current.allConversationEvents).toEqual([
        userMessage,
        systemPrompt,
      ]);
    });

    it("uses renderable events for totalEvents", () => {
      const userMessage = createUserMessage("msg-1");
      const systemPrompt = createSystemPromptEvent("system-1");

      useEventStore.setState({
        events: [userMessage, systemPrompt],
        eventIds: new Set(["msg-1", "system-1"]),
        uiEvents: [userMessage, systemPrompt],
      });

      const { result } = renderHook(() => useFilteredEvents());
      expect(result.current.totalEvents).toBe(1);
    });
  });

  describe("hasSubstantiveAgentActions", () => {
    it("returns false when no events exist", () => {
      const { result } = renderHook(() => useFilteredEvents());
      expect(result.current.hasSubstantiveAgentActions).toBe(false);
    });

    it("returns false when only user events exist", () => {
      const userMessage = createUserMessage("msg-1");

      useEventStore.setState({
        events: [userMessage],
        eventIds: new Set(["msg-1"]),
        uiEvents: [userMessage],
      });

      const { result } = renderHook(() => useFilteredEvents());
      expect(result.current.hasSubstantiveAgentActions).toBe(false);
    });

    it("returns false when only system prompt events exist", () => {
      const systemPrompt = createSystemPromptEvent("system-1");

      useEventStore.setState({
        events: [systemPrompt],
        eventIds: new Set(["system-1"]),
        uiEvents: [systemPrompt],
      });

      const { result } = renderHook(() => useFilteredEvents());
      expect(result.current.hasSubstantiveAgentActions).toBe(false);
    });

    it("returns true when agent action events exist", () => {
      const agentAction = createAgentAction("action-1");

      useEventStore.setState({
        events: [agentAction],
        eventIds: new Set(["action-1"]),
        uiEvents: [agentAction],
      });

      const { result } = renderHook(() => useFilteredEvents());
      expect(result.current.hasSubstantiveAgentActions).toBe(true);
    });
  });

  describe("userEventsExist", () => {
    it("returns false when no events exist", () => {
      const { result } = renderHook(() => useFilteredEvents());
      expect(result.current.userEventsExist).toBe(false);
    });

    it("returns true when user events exist", () => {
      const userMessage = createUserMessage("msg-1");

      useEventStore.setState({
        events: [userMessage],
        eventIds: new Set(["msg-1"]),
        uiEvents: [userMessage],
      });

      const { result } = renderHook(() => useFilteredEvents());
      expect(result.current.conversationUserEventsExist).toBe(true);
      expect(result.current.userEventsExist).toBe(true);
    });
  });

  describe("empty store", () => {
    it("returns empty arrays and false flags for empty store", () => {
      const { result } = renderHook(() => useFilteredEvents());

      expect(result.current.renderableEvents).toEqual([]);
      expect(result.current.allConversationEvents).toEqual([]);
      expect(result.current.totalEvents).toBe(0);
      expect(result.current.hasSubstantiveAgentActions).toBe(false);
      expect(result.current.conversationUserEventsExist).toBe(false);
      expect(result.current.userEventsExist).toBe(false);
    });
  });
});
