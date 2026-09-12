import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { TaskItem } from "#/components/conversation-events/chat/task-tracking/task-item";
import type { TaskItem as TaskItemData } from "#/types/agent-server/core/base/common";
import { I18nKey } from "#/i18n/declaration";

vi.mock("react-i18next", () => ({
  useTranslation: (namespace?: string) => ({
    t: (key: string) =>
      namespace === "openhands" ? key : `missing-namespace:${key}`,
  }),
}));

vi.mock("#/icons/u-circle.svg?react", () => ({
  default: ({ className }: { className?: string }) => (
    <svg data-testid="todo-icon" className={className} />
  ),
}));

vi.mock("#/icons/u-check-circle-half.svg?react", () => ({
  default: ({ className }: { className?: string }) => (
    <svg data-testid="in-progress-icon" className={className} />
  ),
}));

vi.mock("#/icons/u-check-circle.svg?react", () => ({
  default: ({ className }: { className?: string }) => (
    <svg data-testid="done-icon" className={className} />
  ),
}));

function createTask(overrides: Partial<TaskItemData> = {}): TaskItemData {
  return {
    title: "Review the implementation",
    notes: "",
    status: "todo",
    ...overrides,
  };
}

describe("conversation task status display", () => {
  it.each([
    ["todo", "todo-icon", false],
    ["in_progress", "in-progress-icon", false],
    ["done", "done-icon", true],
  ] as const)(
    "renders the %s status with its matching icon and title treatment",
    (status, iconTestId, isMuted) => {
      const task = createTask({ status, title: `${status} task` });

      const { container } = render(<TaskItem task={task} />);

      expect(container.querySelector('[data-name="item"]')).toBeInTheDocument();
      const icon = screen.getByTestId(iconTestId);
      expect(icon).toBeInTheDocument();
      const title = screen.getByText(`${status} task`);
      expect(title).toHaveClass("text-[12px]");

      if (isMuted) {
        expect(icon).toHaveClass("text-[var(--oh-muted)]");
        expect(title).toHaveClass("text-[var(--oh-muted)]");
      } else {
        expect(icon).toHaveClass("text-[#ffffff]");
        expect(title).toHaveClass("text-white");
        expect(title).not.toHaveClass("text-[var(--oh-muted)]");
      }
    },
  );

  it("shows translated notes beneath the task title", () => {
    render(
      <TaskItem
        task={createTask({
          title: "Document the behavior",
          notes: "Include boundary cases",
        })}
      />,
    );

    expect(screen.getByText("Document the behavior")).toBeInTheDocument();
    expect(
      screen.getByText(
        `${I18nKey.TASK_TRACKING_OBSERVATION$TASK_NOTES}: Include boundary cases`,
      ),
    ).toBeInTheDocument();
  });

  it("omits the notes row when notes are empty", () => {
    render(<TaskItem task={createTask({ notes: "" })} />);

    expect(
      screen.queryByText(
        new RegExp(I18nKey.TASK_TRACKING_OBSERVATION$TASK_NOTES),
      ),
    ).not.toBeInTheDocument();
  });

  it("uses the todo presentation for an unknown runtime status", () => {
    const task = createTask({
      status: "blocked" as TaskItemData["status"],
    });

    render(<TaskItem task={task} />);

    expect(screen.getByTestId("todo-icon")).toHaveClass("text-[#ffffff]");
    expect(screen.getByText("Review the implementation")).not.toHaveClass(
      "text-[var(--oh-muted)]",
    );
  });

  it("updates the status icon when task progress changes", () => {
    const { rerender } = render(
      <TaskItem task={createTask({ status: "todo" })} />,
    );
    expect(screen.getByTestId("todo-icon")).toBeInTheDocument();

    rerender(<TaskItem task={createTask({ status: "in_progress" })} />);

    expect(screen.queryByTestId("todo-icon")).not.toBeInTheDocument();
    expect(screen.getByTestId("in-progress-icon")).toBeInTheDocument();
  });
});
