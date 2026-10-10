import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { I18nKey } from "#/i18n/declaration";
import type { ObservationEvent } from "#/types/agent-server/core";
import type { TaskItem as TaskItemType } from "#/types/agent-server/core/base/common";
import type { TaskTrackerObservation } from "#/types/agent-server/core/base/observation";
import { TaskTrackingObservationContent } from "./task-tracking-observation-content";

const createTask = (overrides: Partial<TaskItemType> = {}): TaskItemType => ({
  title: "Inspect the behavior",
  notes: "",
  status: "todo",
  ...overrides,
});

const createTaskTrackingEvent = (
  observationOverrides: Partial<TaskTrackerObservation> = {},
): ObservationEvent<TaskTrackerObservation> => ({
  id: "observation-1",
  timestamp: "2026-07-13T12:00:00.000Z",
  source: "environment",
  tool_name: "task_tracker",
  tool_call_id: "tool-call-1",
  action_id: "action-1",
  observation: {
    kind: "TaskTrackerObservation",
    content: "Current plan",
    command: "plan",
    task_list: [createTask()],
    ...observationOverrides,
  },
});

describe("task tracking observations", () => {
  it("shows a non-empty plan", () => {
    const event = createTaskTrackingEvent({
      task_list: [
        createTask({ title: "Inspect the behavior" }),
        createTask({ title: "Verify the result", status: "done" }),
      ],
    });

    render(<TaskTrackingObservationContent event={event} />);

    expect(screen.getByText(I18nKey.COMMON$TASKS)).toBeInTheDocument();
    expect(screen.getByText("Inspect the behavior")).toBeInTheDocument();
    expect(screen.getByText("Verify the result")).toBeInTheDocument();
  });

  it("hides task data emitted by commands other than plan", () => {
    const event = createTaskTrackingEvent({ command: "update" });

    render(<TaskTrackingObservationContent event={event} />);

    expect(screen.queryByText(I18nKey.COMMON$TASKS)).not.toBeInTheDocument();
    expect(screen.queryByText("Inspect the behavior")).not.toBeInTheDocument();
  });

  it("hides an empty plan", () => {
    const event = createTaskTrackingEvent({ task_list: [] });

    render(<TaskTrackingObservationContent event={event} />);

    expect(screen.queryByText(I18nKey.COMMON$TASKS)).not.toBeInTheDocument();
  });
});
