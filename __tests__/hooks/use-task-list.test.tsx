import React from "react";
import { describe, expect, it, beforeEach } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { NavigationProvider } from "#/context/navigation-context";
import { useTaskList } from "#/hooks/use-task-list";
import { useEventStore } from "#/stores/use-event-store";
import type { OHEvent } from "#/stores/use-event-store";
import type { MessageEvent } from "#/types/agent-server/core";
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

function createTaskTrackerObservation(
  id: string,
  command: string,
  taskList: Array<{
    title: string;
    notes: string;
    status: "todo" | "in_progress" | "done";
  }>,
): OHEvent {
  return {
    id,
    timestamp: `2025-07-01T00:00:0${id}Z`,
    source: "environment",
    tool_name: "task_tracker",
    tool_call_id: `call_${id}`,
    action_id: `action_${id}`,
    observation: {
      kind: "TaskTrackerObservation",
      content: "Task list updated",
      command,
      task_list: taskList,
    },
  } as OHEvent;
}

function createUserMessage(id: string): MessageEvent {
  return {
    id,
    timestamp: `2025-07-01T00:00:0${id}Z`,
    source: "user",
    llm_message: {
      role: "user",
      content: [{ type: "text", text: "Hello" }],
    },
    activated_microagents: [],
    extended_content: [],
  };
}

beforeEach(() => {
  useEventStore.getState().clearEvents();
});

describe("useTaskList", () => {
  it("returns empty taskList and hasTaskList=false when no events exist", () => {
    const { result } = renderHook(() => useTaskList(), { wrapper });

    expect(result.current.taskList).toEqual([]);
    expect(result.current.hasTaskList).toBe(false);
  });

  it("returns empty taskList when no task tracking observations exist", () => {
    const event = createUserMessage("1");
    seedConversationEvents("test-conversation-id", [event], []);

    const { result } = renderHook(() => useTaskList(), { wrapper });

    expect(result.current.taskList).toEqual([]);
    expect(result.current.hasTaskList).toBe(false);
  });

  it('returns the task list from a TaskTrackerObservation with command="plan"', () => {
    const tasks = [
      { title: "First task", notes: "", status: "todo" as const },
      {
        title: "Second task",
        notes: "some note",
        status: "in_progress" as const,
      },
    ];
    const event = createTaskTrackerObservation("1", "plan", tasks);

    seedConversationEvents("test-conversation-id", [event], [event]);

    const { result } = renderHook(() => useTaskList(), { wrapper });

    expect(result.current.taskList).toEqual([
      { id: "1", title: "First task", notes: undefined, status: "todo" },
      {
        id: "2",
        title: "Second task",
        notes: "some note",
        status: "in_progress",
      },
    ]);
    expect(result.current.hasTaskList).toBe(true);
  });

  it('ignores TaskTrackerObservation with command !== "plan"', () => {
    const tasks = [{ title: "First task", notes: "", status: "todo" as const }];
    const event = createTaskTrackerObservation("1", "view", tasks);

    seedConversationEvents("test-conversation-id", [event], [event]);

    const { result } = renderHook(() => useTaskList(), { wrapper });

    expect(result.current.taskList).toEqual([]);
    expect(result.current.hasTaskList).toBe(false);
  });

  it("returns the latest task list when multiple plan events exist", () => {
    const earlyTasks = [
      { title: "First task", notes: "", status: "todo" as const },
    ];
    const lateTasks = [
      { title: "First task", notes: "", status: "done" as const },
      { title: "New task", notes: "wip", status: "in_progress" as const },
    ];

    const event1 = createTaskTrackerObservation("1", "plan", earlyTasks);
    const event2 = createTaskTrackerObservation("2", "plan", lateTasks);

    seedConversationEvents("test-conversation-id", [event1, event2], [event1, event2]);

    const { result } = renderHook(() => useTaskList(), { wrapper });

    expect(result.current.taskList).toEqual([
      { id: "1", title: "First task", notes: undefined, status: "done" },
      { id: "2", title: "New task", notes: "wip", status: "in_progress" },
    ]);
    expect(result.current.hasTaskList).toBe(true);
  });

  it("updates when new events are added to the store", () => {
    const { result } = renderHook(() => useTaskList(), { wrapper });

    expect(result.current.hasTaskList).toBe(false);

    const tasks = [{ title: "New task", notes: "", status: "todo" as const }];
    const event = createTaskTrackerObservation("1", "plan", tasks);

    act(() => {
      seedConversationEvents("test-conversation-id", [event], [event]);
    });

    expect(result.current.taskList).toEqual([
      { id: "1", title: "New task", notes: undefined, status: "todo" },
    ]);
    expect(result.current.hasTaskList).toBe(true);
  });

  it("returns hasTaskList=false when the latest plan has an empty task list", () => {
    const event = createTaskTrackerObservation("1", "plan", []);

    seedConversationEvents("test-conversation-id", [event], [event]);

    const { result } = renderHook(() => useTaskList(), { wrapper });

    expect(result.current.taskList).toEqual([]);
    expect(result.current.hasTaskList).toBe(false);
  });
});
