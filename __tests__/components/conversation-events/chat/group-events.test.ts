import { describe, it, expect } from "vitest";
import {
  groupEvents,
  isGroupableEvent,
  EVENT_GROUP_MIN_SIZE,
} from "#/components/conversation-events/chat/group-events";
import {
  ActionEvent,
  AgentErrorEvent,
  HookExecutionEvent,
  MessageEvent,
  ObservationEvent,
  SecurityRisk,
} from "#/types/agent-server/core";
import { TextContent } from "#/types/agent-server/core/base/common";
import {
  ExecuteBashAction,
  FinishAction,
  ThinkAction,
} from "#/types/agent-server/core/base/action";
import {
  ExecuteBashObservation,
  PlanningFileEditorObservation,
  TaskTrackerObservation,
} from "#/types/agent-server/core/base/observation";

const makeBashAction = (
  id: string,
  thought: TextContent[] = [],
): ActionEvent<ExecuteBashAction> => ({
  id,
  timestamp: new Date().toISOString(),
  source: "agent",
  thought,
  thinking_blocks: [],
  action: {
    kind: "ExecuteBashAction",
    command: `echo ${id}`,
    is_input: false,
    timeout: null,
    reset: false,
  },
  tool_name: "execute_bash",
  tool_call_id: `call_${id}`,
  tool_call: {
    id: `call_${id}`,
    type: "function",
    function: {
      name: "execute_bash",
      arguments: JSON.stringify({ command: `echo ${id}` }),
    },
  },
  llm_response_id: `response_${id}`,
  security_risk: SecurityRisk.UNKNOWN,
});

const makeBashObservation = (
  id: string,
  actionId: string,
): ObservationEvent<ExecuteBashObservation> => ({
  id,
  timestamp: new Date().toISOString(),
  source: "environment",
  tool_name: "execute_bash",
  tool_call_id: `call_${actionId}`,
  action_id: actionId,
  observation: {
    kind: "ExecuteBashObservation",
    content: [{ type: "text", text: "ok" }],
    command: `echo ${actionId}`,
    exit_code: 0,
    error: false,
    timeout: false,
    metadata: {} as never,
  },
});

const makeUserMessage = (id: string): MessageEvent => ({
  id,
  timestamp: new Date().toISOString(),
  source: "user",
  llm_message: {
    role: "user",
    content: [{ type: "text", text: "hi" }],
  },
  activated_microagents: [],
  extended_content: [],
});

const makeFinishAction = (id: string): ActionEvent<FinishAction> => ({
  ...makeBashAction(id),
  action: { kind: "FinishAction", message: "all done" },
});

const makeThinkAction = (id: string): ActionEvent<ThinkAction> => ({
  ...makeBashAction(id),
  action: { kind: "ThinkAction", thought: "thinking" },
});

const makePlanObs = (
  id: string,
  actionId: string,
): ObservationEvent<PlanningFileEditorObservation> => ({
  ...makeBashObservation(id, actionId),
  observation: {
    kind: "PlanningFileEditorObservation",
    content: [{ type: "text", text: "plan" }],
  } as PlanningFileEditorObservation,
});

const makeTaskTrackerObservation = (
  id: string,
  actionId: string,
): ObservationEvent<TaskTrackerObservation> => ({
  ...makeBashObservation(id, actionId),
  observation: {
    kind: "TaskTrackerObservation",
    content: "task list",
    command: "plan",
    task_list: [{ title: "Task", notes: "Notes", status: "todo" }],
  },
});

const makeAgentErrorEvent = (id: string): AgentErrorEvent => ({
  id,
  timestamp: new Date().toISOString(),
  source: "agent",
  tool_name: "execute_bash",
  tool_call_id: `call_${id}`,
  error: "boom",
});

const makeHookExecutionEvent = (id: string): HookExecutionEvent => ({
  id,
  timestamp: new Date().toISOString(),
  kind: "HookExecutionEvent",
  source: "hook",
  hook_event_type: "PreToolUse",
  hook_command: "echo hook",
  success: true,
  blocked: false,
  exit_code: 0,
  reason: null,
  tool_name: "execute_bash",
  action_id: "a1",
  message_id: null,
  stdout: "",
  stderr: "",
  error: null,
  additional_context: null,
  hook_input: null,
});

describe("isGroupableEvent", () => {
  it("groups regular action events", () => {
    expect(isGroupableEvent(makeBashAction("a1"))).toBe(true);
  });

  it("groups regular observation events", () => {
    expect(isGroupableEvent(makeBashObservation("o1", "a1"))).toBe(true);
  });

  it("does not group user messages", () => {
    expect(isGroupableEvent(makeUserMessage("m1"))).toBe(false);
  });

  it("does not group FinishAction", () => {
    expect(isGroupableEvent(makeFinishAction("a1"))).toBe(false);
  });

  it("does not group ThinkAction", () => {
    expect(isGroupableEvent(makeThinkAction("a1"))).toBe(false);
  });

  it("does not group PlanningFileEditorObservation", () => {
    expect(isGroupableEvent(makePlanObs("o1", "a1"))).toBe(false);
  });

  it("does not group TaskTrackerObservation", () => {
    expect(isGroupableEvent(makeTaskTrackerObservation("o1", "a1"))).toBe(false);
  });

  it("does not group AgentErrorEvent", () => {
    expect(isGroupableEvent(makeAgentErrorEvent("e1"))).toBe(false);
  });

  it("does not group HookExecutionEvent", () => {
    expect(isGroupableEvent(makeHookExecutionEvent("h1"))).toBe(false);
  });
});

describe("groupEvents", () => {
  it("returns an empty list for no events", () => {
    expect(groupEvents([])).toEqual([]);
  });

  it("requires minSize to be at least 1", () => {
    expect(() => groupEvents([], 0)).toThrow("minSize must be at least 1");
  });

  it("emits singles when there are not enough actions in a row", () => {
    const events = [
      makeUserMessage("m1"),
      makeBashObservation("o1", "a1"),
      makeUserMessage("m2"),
    ];

    // 1 < EVENT_GROUP_MIN_SIZE (2), so the lone observation stays a single.
    const result = groupEvents(events);
    expect(result.every((item) => item.kind === "single")).toBe(true);
    expect(result).toHaveLength(3);
  });

  it("groups even a pair of consecutive actions", () => {
    const events = [
      makeUserMessage("m1"),
      makeBashObservation("o1", "a1"),
      makeBashObservation("o2", "a2"),
      makeUserMessage("m2"),
    ];

    // 2 == EVENT_GROUP_MIN_SIZE (2), so the pair folds into a group.
    const result = groupEvents(events);
    expect(result).toHaveLength(3);
    expect(result[0]).toMatchObject({ kind: "single", index: 0 });
    expect(result[1]).toMatchObject({ kind: "group", startIndex: 1 });
    if (result[1].kind === "group") {
      expect(result[1].events).toHaveLength(2);
    }
    expect(result[2]).toMatchObject({ kind: "single", index: 3 });
  });

  it("groups runs of >= EVENT_GROUP_MIN_SIZE events", () => {
    const events = [
      makeUserMessage("m1"),
      makeBashObservation("o1", "a1"),
      makeBashObservation("o2", "a2"),
      makeBashObservation("o3", "a3"),
      makeBashObservation("o4", "a4"),
      makeUserMessage("m2"),
    ];

    const result = groupEvents(events);
    // user, group(4), user
    expect(result).toHaveLength(3);
    expect(result[0]).toMatchObject({ kind: "single", index: 0 });
    expect(result[1]).toMatchObject({ kind: "group", startIndex: 1 });
    if (result[1].kind === "group") {
      expect(result[1].events).toHaveLength(4);
    }
    expect(result[2]).toMatchObject({ kind: "single", index: 5 });
  });

  it("includes a still-pending action in the group", () => {
    // Last action hasn't been replaced by an observation yet -> the group
    // should still include it so the header can show a running indicator.
    const events = [
      makeBashObservation("o1", "a1"),
      makeBashObservation("o2", "a2"),
      makeBashAction("a3"),
    ];

    const result = groupEvents(events);
    expect(result).toHaveLength(1);
    expect(result[0].kind).toBe("group");
    if (result[0].kind === "group") {
      expect(result[0].events).toHaveLength(3);
    }
  });

  it("breaks the group when a non-groupable event interrupts", () => {
    const events = [
      makeBashObservation("o1", "a1"),
      makeBashObservation("o2", "a2"),
      makeBashObservation("o3", "a3"),
      makeUserMessage("m1"),
      makeBashObservation("o4", "a4"),
      makeBashObservation("o5", "a5"),
      makeBashObservation("o6", "a6"),
    ];

    const result = groupEvents(events);
    expect(result).toHaveLength(3);
    expect(result[0].kind).toBe("group");
    expect(result[1].kind).toBe("single");
    expect(result[2].kind).toBe("group");
  });

  it("respects a custom minSize", () => {
    const events = [
      makeBashObservation("o1", "a1"),
      makeBashObservation("o2", "a2"),
    ];

    const grouped = groupEvents(events, 2);
    expect(grouped).toHaveLength(1);
    expect(grouped[0].kind).toBe("group");
  });

  it("hoists an action's thought into a separate item and starts a new group", () => {
    // First three actions have no thought, then a fourth action carries a
    // thought, then two more actions follow. The thought should break the
    // first group and open a new one starting with the thought-bearing
    // action.
    const a1 = makeBashAction("a1");
    const a2 = makeBashAction("a2");
    const a3 = makeBashAction("a3");
    const a4 = makeBashAction("a4", [
      { type: "text", text: "Let me check tests for the new conversation button:" },
    ]);
    const a5 = makeBashAction("a5");
    const a6 = makeBashAction("a6");

    const events = [
      makeBashObservation("o1", "a1"),
      makeBashObservation("o2", "a2"),
      makeBashObservation("o3", "a3"),
      makeBashObservation("o4", "a4"),
      makeBashObservation("o5", "a5"),
      makeBashObservation("o6", "a6"),
    ];
    const allEvents = [a1, a2, a3, ...events.slice(0, 3), a4, a5, a6, ...events.slice(3)];

    const result = groupEvents(events, undefined, allEvents);

    // group(o1..o3), thought(a4), group(o4..o6)
    expect(result).toHaveLength(3);
    expect(result[0].kind).toBe("group");
    expect(result[1].kind).toBe("thought");
    expect(result[2].kind).toBe("group");

    if (result[0].kind === "group") {
      expect(result[0].events).toHaveLength(3);
    }
    if (result[1].kind === "thought") {
      expect(result[1].action.id).toBe("a4");
    }
    if (result[2].kind === "group") {
      expect(result[2].events).toHaveLength(3);
    }
  });

  it("hoists thoughts attached to action events that haven't been observed yet", () => {
    const events = [
      makeBashAction("a1"),
      makeBashAction("a2"),
      makeBashAction("a3", [{ type: "text", text: "thinking out loud" }]),
    ];

    const result = groupEvents(events);

    // a1 + a2 group together (>= min size of 2), then the thought is hoisted
    // out before a3, leaving a3 as a single.
    expect(result.map((item) => item.kind)).toEqual([
      "group",
      "thought",
      "single",
    ]);
  });

  it("does not emit the same thought twice when both action and observation are present", () => {
    const action = makeBashAction("a1", [
      { type: "text", text: "thinking out loud" },
    ]);
    const observation = makeBashObservation("o1", "a1");

    const result = groupEvents(
      [action, observation],
      undefined,
      [action, observation],
    );

    expect(result.map((item) => item.kind)).toEqual(["thought", "group"]);
    if (result[1].kind === "group") {
      expect(result[1].events).toEqual([action, observation]);
    }
  });

  it("ignores empty thoughts", () => {
    const events = [
      makeBashAction("a1", [{ type: "text", text: "   " }]),
      makeBashAction("a2"),
      makeBashAction("a3"),
    ];

    const result = groupEvents(events);

    // Whitespace-only thought should not break the group.
    expect(result).toHaveLength(1);
    expect(result[0].kind).toBe("group");
  });

  it("does not hoist a ThinkAction's thought (it has its own rendering)", () => {
    const events = [
      makeBashAction("a1"),
      makeBashAction("a2"),
      makeBashAction("a3"),
      makeThinkAction("t1"),
      makeBashAction("a4"),
      makeBashAction("a5"),
      makeBashAction("a6"),
    ];

    const result = groupEvents(events);

    // ThinkAction breaks the run as a single (it isn't groupable), but no
    // additional "thought" item should be emitted for it because the
    // ThinkAction's body already renders the thought.
    expect(result.map((item) => item.kind)).toEqual([
      "group",
      "single",
      "group",
    ]);
  });
});
