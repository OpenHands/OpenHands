import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import TaskListTab from "#/routes/task-list-tab";
import { useEventStore } from "#/stores/use-event-store";
import type { TaskTrackingObservation } from "#/types/core/observations";

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

function createTaskTrackingObservation(
  id: number,
  tasks: TaskTrackingObservation["extras"]["task_list"],
): TaskTrackingObservation {
  return {
    id,
    source: "agent",
    observation: "task_tracking",
    message: "Task tracking update",
    timestamp: `2025-07-01T00:00:0${id}Z`,
    cause: 0,
    content: "",
    extras: {
      command: "plan",
      task_list: tasks,
    },
  };
}

function setTasks(tasks: TaskTrackingObservation["extras"]["task_list"]) {
  const event = createTaskTrackingObservation(1, tasks);
  useEventStore.setState({
    events: [event],
    eventIds: new Set([1]),
    uiEvents: [event],
  });
}

beforeEach(() => {
  useEventStore.setState({
    events: [],
    eventIds: new Set(),
    uiEvents: [],
  });
});

describe("TaskListTab", () => {
  it("renders empty state with icon and message when there are no tasks", () => {
    const { container } = render(<TaskListTab />);

    expect(screen.getByText("No tasks yet")).toBeInTheDocument();
    // Empty state should show the check-circle icon (rendered as SVG)
    expect(container.querySelector("svg")).toBeInTheDocument();
  });

  it("renders empty state message using Text component (span)", () => {
    render(<TaskListTab />);

    const message = screen.getByText("No tasks yet");
    expect(message.tagName).toBe("SPAN");
  });

  it("renders task items when tasks exist", () => {
    setTasks([
      { id: "1", title: "Implement feature", status: "todo" },
      { id: "2", title: "Write tests", status: "in_progress" },
      { id: "3", title: "Deploy", status: "done" },
    ]);

    const { container } = render(<TaskListTab />);

    expect(screen.getByText("Implement feature")).toBeInTheDocument();
    expect(screen.getByText("Write tests")).toBeInTheDocument();
    expect(screen.getByText("Deploy")).toBeInTheDocument();

    const taskItems = container.querySelectorAll('[data-name="item"]');
    expect(taskItems).toHaveLength(3);
  });

  it("does not display task IDs", () => {
    setTasks([
      { id: "task-1", title: "First task", status: "todo" },
    ]);

    render(<TaskListTab />);

    expect(screen.queryByText(/task-1/)).not.toBeInTheDocument();
  });

  it("highlights in_progress tasks with a background", () => {
    setTasks([
      { id: "1", title: "Todo task", status: "todo" },
      { id: "2", title: "Active task", status: "in_progress" },
      { id: "3", title: "Done task", status: "done" },
    ]);

    render(<TaskListTab />);

    // Find each task item via its text, then check the wrapper div
    const activeItem = screen.getByText("Active task").closest("[data-name]");
    const activeWrapper = activeItem?.parentElement;
    expect(activeWrapper?.className).toContain("bg-[#2D3039]");

    const todoItem = screen.getByText("Todo task").closest("[data-name]");
    expect(todoItem?.parentElement?.className).not.toContain("bg-[#2D3039]");

    const doneItem = screen.getByText("Done task").closest("[data-name]");
    expect(doneItem?.parentElement?.className).not.toContain("bg-[#2D3039]");
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

    render(<TaskListTab />);

    expect(screen.getByText("Notes: Important note")).toBeInTheDocument();
    expect(screen.getAllByText(/^Notes:/)).toHaveLength(1);
  });

  it("uses the latest plan event when multiple exist", () => {
    const event1 = createTaskTrackingObservation(1, [
      { id: "1", title: "Old task", status: "todo" },
    ]);
    const event2 = createTaskTrackingObservation(2, [
      { id: "1", title: "Updated task", status: "done" },
      { id: "2", title: "New task", status: "in_progress" },
    ]);

    useEventStore.setState({
      events: [event1, event2],
      eventIds: new Set([1, 2]),
      uiEvents: [event1, event2],
    });

    render(<TaskListTab />);

    expect(screen.queryByText("Old task")).not.toBeInTheDocument();
    expect(screen.getByText("Updated task")).toBeInTheDocument();
    expect(screen.getByText("New task")).toBeInTheDocument();
  });

  it("renders as a scrollable main element when tasks exist", () => {
    setTasks([{ id: "1", title: "A task", status: "todo" }]);

    render(<TaskListTab />);

    const main = screen.getByRole("main");
    expect(main).toBeInTheDocument();
  });
});
