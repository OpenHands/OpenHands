import { useMemo } from "react";
import { useEventStore } from "#/stores/use-event-store";
import type { OHEvent } from "#/stores/use-event-store";
import { isTaskTrackingObservation } from "#/types/core/guards";
import type { OpenHandsParsedEvent } from "#/types/core";
import { isObservationEvent } from "#/types/v1/type-guards";
import type { OpenHandsEvent } from "#/types/v1/core";
import type { TaskTrackerObservation } from "#/types/v1/core/base/observation";
import type { ObservationEvent } from "#/types/v1/core/events/observation-event";

export interface TaskListItem {
  id: string;
  title: string;
  status: "todo" | "in_progress" | "done";
  notes?: string;
}

function getTaskListFromEvent(event: OHEvent): TaskListItem[] | null {
  // v0 event format: observation is a string "task_tracking"
  const v0 = event as OpenHandsParsedEvent;
  if (isTaskTrackingObservation(v0) && v0.extras.command === "plan") {
    return v0.extras.task_list.map((t) => ({
      id: t.id,
      title: t.title,
      status: t.status,
      notes: t.notes,
    }));
  }

  // v1 event format: observation is an object with kind "TaskTrackerObservation"
  const v1 = event as OpenHandsEvent;
  if (
    isObservationEvent(v1) &&
    v1.observation.kind === "TaskTrackerObservation"
  ) {
    const obs = (v1 as ObservationEvent<TaskTrackerObservation>).observation;
    if (obs.command === "plan") {
      return obs.task_list.map((t, i) => ({
        id: String(i + 1),
        title: t.title,
        status: t.status,
        notes: t.notes || undefined,
      }));
    }
  }

  return null;
}

export function useTaskList() {
  const events = useEventStore((state) => state.events);

  return useMemo(() => {
    // Iterate in reverse to find the latest TaskTrackingObservation with command="plan"
    for (let i = events.length - 1; i >= 0; i -= 1) {
      const taskList = getTaskListFromEvent(events[i]);
      if (taskList) {
        return { taskList, hasTaskList: taskList.length > 0 };
      }
    }

    return { taskList: [] as TaskListItem[], hasTaskList: false };
  }, [events]);
}
