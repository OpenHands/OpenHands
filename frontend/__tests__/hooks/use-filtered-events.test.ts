import { describe, expect, it, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useFilteredEvents } from "#/hooks/use-filtered-events";
import { useEventStore } from "#/stores/use-event-store";
import type { OpenHandsAction } from "#/types/core/actions";
import type { ActionEvent, MessageEvent } from "#/types/v1/core";
import { SecurityRisk } from "#/types/v1/core";

// --- V0 event factories ---

function createV0UserMessage(id: number): OpenHandsAction {
  return {
    id,
    source: "user",
    action: "message",
    args: { content: `User message ${id}`, image_urls: [], file_urls: [] },
    message: `User message ${id}`,
    timestamp: `2025-07-01T00:00:0${id}Z`,
  };
}

function createV0AgentMessage(id: number): OpenHandsAction {
  return {
    id,
    source: "agent",
    action: "message",
    args: {
      thought: `Agent thought ${id}`,
      image_urls: null,
      file_urls: [],
      wait_for_response: true,
    },
    message: `Agent response ${id}`,
    timestamp: `2025-07-01T00:00:0${id}Z`,
  };
}

function createV0SystemEvent(id: number): OpenHandsAction {
  return {
    id,
    source: "environment",
    action: "system",
    args: {
      content: "source .openhands/setup.sh",
      tools: null,
      openhands_version: null,
      agent_class: null,
    },
    message: "Running setup script",
    timestamp: `2025-07-01T00:00:0${id}Z`,
  };
}

// --- V1 event factories ---

function createV1UserMessage(id: string): MessageEvent {
  return {
    id,
    timestamp: "2025-07-01T00:00:01Z",
    source: "user",
    llm_message: {
      role: "user",
      content: [{ type: "text", text: `User message ${id}` }],
    },
    activated_microagents: [],
    extended_content: [],
  };
}

function createV1AgentAction(id: string): ActionEvent {
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

beforeEach(() => {
  // Reset the event store before each test
  useEventStore.setState({
    events: [],
    eventIds: new Set(),
    uiEvents: [],
  });
});

describe("useFilteredEvents", () => {
  describe("referential stability", () => {
    it("returns the same v0Events reference when storeEvents has not changed", () => {
      const v0Event = createV0UserMessage(1);
      useEventStore.setState({
        events: [v0Event],
        eventIds: new Set([1]),
        uiEvents: [v0Event],
      });

      const { result, rerender } = renderHook(() => useFilteredEvents());
      const firstV0Events = result.current.v0Events;

      // Rerender without changing the store
      rerender();

      expect(result.current.v0Events).toBe(firstV0Events);
    });

    it("returns the same v1UiEvents reference when uiEvents has not changed", () => {
      const v1Event = createV1UserMessage("msg-1");
      useEventStore.setState({
        events: [v1Event],
        eventIds: new Set(["msg-1"]),
        uiEvents: [v1Event],
      });

      const { result, rerender } = renderHook(() => useFilteredEvents());
      const firstV1UiEvents = result.current.v1UiEvents;

      rerender();

      expect(result.current.v1UiEvents).toBe(firstV1UiEvents);
    });

    it("returns the same v1FullEvents reference when storeEvents has not changed", () => {
      const v1Event = createV1UserMessage("msg-1");
      useEventStore.setState({
        events: [v1Event],
        eventIds: new Set(["msg-1"]),
        uiEvents: [v1Event],
      });

      const { result, rerender } = renderHook(() => useFilteredEvents());
      const firstV1FullEvents = result.current.v1FullEvents;

      rerender();

      expect(result.current.v1FullEvents).toBe(firstV1FullEvents);
    });

    it("returns a new v0Events reference when storeEvents changes", () => {
      const v0Event1 = createV0UserMessage(1);
      useEventStore.setState({
        events: [v0Event1],
        eventIds: new Set([1]),
        uiEvents: [v0Event1],
      });

      const { result } = renderHook(() => useFilteredEvents());
      const firstV0Events = result.current.v0Events;

      // Add a new event to the store (new array reference)
      const v0Event2 = createV0AgentMessage(2);
      act(() => {
        useEventStore.setState({
          events: [v0Event1, v0Event2],
          eventIds: new Set([1, 2]),
          uiEvents: [v0Event1, v0Event2],
        });
      });

      expect(result.current.v0Events).not.toBe(firstV0Events);
      expect(result.current.v0Events).toHaveLength(2);
    });
  });

  describe("V0 event filtering", () => {
    it("filters V0 events through isV0Event, isActionOrObservation, and shouldRenderEvent", () => {
      const userMsg = createV0UserMessage(1);
      const agentMsg = createV0AgentMessage(2);

      useEventStore.setState({
        events: [userMsg, agentMsg],
        eventIds: new Set([1, 2]),
        uiEvents: [userMsg, agentMsg],
      });

      const { result } = renderHook(() => useFilteredEvents());

      expect(result.current.v0Events).toHaveLength(2);
      expect(result.current.v0Events).toContainEqual(userMsg);
      expect(result.current.v0Events).toContainEqual(agentMsg);
    });

    it("excludes V0 system events from v0Events", () => {
      const userMsg = createV0UserMessage(1);
      const systemEvent = createV0SystemEvent(2);

      useEventStore.setState({
        events: [userMsg, systemEvent],
        eventIds: new Set([1, 2]),
        uiEvents: [userMsg, systemEvent],
      });

      const { result } = renderHook(() => useFilteredEvents());

      // System events are filtered out by shouldRenderEvent
      expect(result.current.v0Events).toHaveLength(1);
      expect(result.current.v0Events[0]).toEqual(userMsg);
    });

    it("does not include V1 events in v0Events", () => {
      const v0Event = createV0UserMessage(1);
      const v1Event = createV1UserMessage("msg-1");

      useEventStore.setState({
        events: [v0Event, v1Event],
        eventIds: new Set([1, "msg-1"]),
        uiEvents: [v0Event, v1Event],
      });

      const { result } = renderHook(() => useFilteredEvents());

      expect(result.current.v0Events).toHaveLength(1);
      expect(result.current.v0Events[0]).toEqual(v0Event);
    });
  });

  describe("V1 event filtering", () => {
    it("filters V1 events into v1FullEvents", () => {
      const v1Event = createV1UserMessage("msg-1");

      useEventStore.setState({
        events: [v1Event],
        eventIds: new Set(["msg-1"]),
        uiEvents: [v1Event],
      });

      const { result } = renderHook(() => useFilteredEvents());

      expect(result.current.v1FullEvents).toHaveLength(1);
      expect(result.current.v1FullEvents[0]).toEqual(v1Event);
    });

    it("does not include V0 events in v1FullEvents", () => {
      const v0Event = createV0UserMessage(1);
      const v1Event = createV1UserMessage("msg-1");

      useEventStore.setState({
        events: [v0Event, v1Event],
        eventIds: new Set([1, "msg-1"]),
        uiEvents: [v0Event, v1Event],
      });

      const { result } = renderHook(() => useFilteredEvents());

      expect(result.current.v1FullEvents).toHaveLength(1);
      expect(result.current.v1FullEvents[0]).toEqual(v1Event);
    });
  });

  describe("totalEvents", () => {
    it("returns V0 event count when V0 events exist", () => {
      const v0Event1 = createV0UserMessage(1);
      const v0Event2 = createV0AgentMessage(2);

      useEventStore.setState({
        events: [v0Event1, v0Event2],
        eventIds: new Set([1, 2]),
        uiEvents: [v0Event1, v0Event2],
      });

      const { result } = renderHook(() => useFilteredEvents());
      expect(result.current.totalEvents).toBe(2);
    });

    it("returns 0 when no events exist", () => {
      const { result } = renderHook(() => useFilteredEvents());
      expect(result.current.totalEvents).toBe(0);
    });
  });

  describe("hasSubstantiveAgentActions", () => {
    it("returns false when no events exist", () => {
      const { result } = renderHook(() => useFilteredEvents());
      expect(result.current.hasSubstantiveAgentActions).toBe(false);
    });

    it("returns false when only user events exist (V0)", () => {
      const userMsg = createV0UserMessage(1);

      useEventStore.setState({
        events: [userMsg],
        eventIds: new Set([1]),
        uiEvents: [userMsg],
      });

      const { result } = renderHook(() => useFilteredEvents());
      expect(result.current.hasSubstantiveAgentActions).toBe(false);
    });

    it("returns true when V0 agent message actions exist", () => {
      const agentMsg = createV0AgentMessage(1);

      useEventStore.setState({
        events: [agentMsg],
        eventIds: new Set([1]),
        uiEvents: [agentMsg],
      });

      const { result } = renderHook(() => useFilteredEvents());
      expect(result.current.hasSubstantiveAgentActions).toBe(true);
    });

    it("returns true when V1 agent action events exist", () => {
      const agentAction = createV1AgentAction("action-1");

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

    it("returns true when V0 user events exist", () => {
      const userMsg = createV0UserMessage(1);

      useEventStore.setState({
        events: [userMsg],
        eventIds: new Set([1]),
        uiEvents: [userMsg],
      });

      const { result } = renderHook(() => useFilteredEvents());
      expect(result.current.v0UserEventsExist).toBe(true);
      expect(result.current.userEventsExist).toBe(true);
    });

    it("returns true when V1 user events exist", () => {
      const userMsg = createV1UserMessage("msg-1");

      useEventStore.setState({
        events: [userMsg],
        eventIds: new Set(["msg-1"]),
        uiEvents: [userMsg],
      });

      const { result } = renderHook(() => useFilteredEvents());
      expect(result.current.v1UserEventsExist).toBe(true);
      expect(result.current.userEventsExist).toBe(true);
    });
  });

  describe("empty store", () => {
    it("returns empty arrays and false flags for empty store", () => {
      const { result } = renderHook(() => useFilteredEvents());

      expect(result.current.v0Events).toEqual([]);
      expect(result.current.v1UiEvents).toEqual([]);
      expect(result.current.v1FullEvents).toEqual([]);
      expect(result.current.totalEvents).toBe(0);
      expect(result.current.hasSubstantiveAgentActions).toBe(false);
      expect(result.current.v0UserEventsExist).toBe(false);
      expect(result.current.v1UserEventsExist).toBe(false);
      expect(result.current.userEventsExist).toBe(false);
    });
  });
});
