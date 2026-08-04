import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import TaskListTab from "#/routes/task-list-tab";
import { useEventStore } from "#/stores/use-event-store";
import type { OHEvent } from "#/stores/use-event-store";
import { seedConversationEvents } from "../helpers/seed-conversation-events";
import { NavigationProvider } from "#/context/navigation-context";

// Mock i18n
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => {
      const translations: Record<string, string> = {
        COMMON$NO_TASKS: "No tasks yet",
        TASK_TRACKING_OBSERVATION$TASK_NOTES: "Notes",
      };
      return translations[key] || key;
    },
  }),
}));

type TestTask = {
  id?: string;
  title: string;
  status: "todo" | "in_progress" | "done";
  notes?: string;
};

function createTaskTrackingObservation(id: string, tasks: TestTask[]): OHEvent {
  return {
    id,
    timestamp: `2025-07-01T00:00:0${id}Z`,
    source: "environment",
    tool_name: "task_tracker",
    tool_call_id: `call_${id}`,
    action_id: `action_${id}`,
    observation: {
      kind: "TaskTrackerObservation",
      content: "Task tracking update",
      command: "plan",
      task_list: tasks.map((task) => ({
        title: task.title,
        notes: task.notes ?? "",
        status: task.status,
      })),
    },
  } as OHEvent;
}

function setTasks(tasks: TestTask[]) {
  const event = createTaskTrackingObservation("1", tasks);
  seedConversationEvents("test-conversation-id", [event], [event]);
}

function renderTaskList() {
  return render(
    <NavigationProvider
      value={{
        currentPath: "/conversations/test-conversation-id",
        conversationId: "test-conversation-id",
        isNavigating: false,
        navigate: vi.fn(),
      }}
    >
      <TaskListTab />
    </NavigationProvider>,
  );
}

beforeEach(() => {
  useEventStore.getState().clearEvents();
});

describe("TaskListTab", () => {
  it("renders empty state with icon and message when there are no tasks", () => {
    const { container } = renderTaskList();

    expect(screen.getByText("No tasks yet")).toBeInTheDocument();
    // Empty state should show the check-circle icon (rendered as SVG)
    expect(container.querySelector("svg")).toBeInTheDocument();
  });

  it("renders empty state message in a centered caption", () => {
    renderTaskList();

    const message = screen.getByText("No tasks yet");
    expect(message.tagName).toBe("P");
  });

  it("renders task items when tasks exist", () => {
    setTasks([
      { id: "1", title: "Implement feature", status: "todo" },
      { id: "2", title: "Write tests", status: "in_progress" },
      { id: "3", title: "Deploy", status: "done" },
    ]);

    const { container } = renderTaskList();

    expect(screen.getByText("Implement feature")).toBeInTheDocument();
    expect(screen.getByText("Write tests")).toBeInTheDocument();
    expect(screen.getByText("Deploy")).toBeInTheDocument();

    const taskItems = container.querySelectorAll('[data-name="item"]');
    expect(taskItems).toHaveLength(3);
  });

  it("does not display task IDs", () => {
    setTasks([{ id: "task-1", title: "First task", status: "todo" }]);

    renderTaskList();

    expect(screen.queryByText(/task-1/)).not.toBeInTheDocument();
  });

  it("marks in_progress tasks as active via data-active", () => {
    setTasks([
      { id: "1", title: "Todo task", status: "todo" },
      { id: "2", title: "Active task", status: "in_progress" },
      { id: "3", title: "Done task", status: "done" },
    ]);

    renderTaskList();

    // Find each task item via its text, then check the wrapper div
    const activeWrapper = screen
      .getByText("Active task")
      .closest("[data-name]")?.parentElement;
    expect(activeWrapper).toHaveAttribute("data-active", "true");

    const todoWrapper = screen
      .getByText("Todo task")
      .closest("[data-name]")?.parentElement;
    expect(todoWrapper).toHaveAttribute("data-active", "false");

    const doneWrapper = screen
      .getByText("Done task")
      .closest("[data-name]")?.parentElement;
    expect(doneWrapper).toHaveAttribute("data-active", "false");
  });

  it("displays task notes when present and omits when absent", () => {
    setTasks([
      {
        id: "1",
        title: "Task with notes",
        status: "todo",
        notes: "Important note",
      },
      { id: "2", title: "Task without notes", status: "todo" },
    ]);

    renderTaskList();

    expect(screen.getByText("Notes: Important note")).toBeInTheDocument();
    expect(screen.getAllByText(/^Notes:/)).toHaveLength(1);
  });

  it("uses the latest plan event when multiple exist", () => {
    const event1 = createTaskTrackingObservation("1", [
      { id: "1", title: "Old task", status: "todo" },
    ]);
    const event2 = createTaskTrackingObservation("2", [
      { id: "1", title: "Updated task", status: "done" },
      { id: "2", title: "New task", status: "in_progress" },
    ]);

    seedConversationEvents(
      "test-conversation-id",
      [event1, event2],
      [event1, event2],
    );

    renderTaskList();

    expect(screen.queryByText("Old task")).not.toBeInTheDocument();
    expect(screen.getByText("Updated task")).toBeInTheDocument();
    expect(screen.getByText("New task")).toBeInTheDocument();
  });

  it("renders as a scrollable main element when tasks exist", () => {
    setTasks([{ id: "1", title: "A task", status: "todo" }]);

    renderTaskList();

    const main = screen.getByRole("main");
    expect(main).toBeInTheDocument();
  });
});
