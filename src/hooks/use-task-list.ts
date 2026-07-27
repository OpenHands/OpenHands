import { useMemo } from "react";
import { useEventStore } from "#/stores/use-event-store";
import { isObservationEvent } from "#/types/agent-server/type-guards";
import type { TaskTrackerObservation } from "#/types/agent-server/core/base/observation";
import type { ObservationEvent } from "#/types/agent-server/core/events/observation-event";

export interface TaskListItem {
  id: string;
  title: string;
  status: "todo" | "in_progress" | "done";
  notes?: string;
}

function getTaskListFromEvent(
  event: ReturnType<typeof useEventStore.getState>["events"][number],
): TaskListItem[] | null {
  if (
    isObservationEvent(event) &&
    event.observation.kind === "TaskTrackerObservation"
  ) {
    const obs = (event as ObservationEvent<TaskTrackerObservation>).observation;
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
    for (let i = events.length - 1; i >= 0; i -= 1) {
      const taskList = getTaskListFromEvent(events[i]);
      if (taskList) {
        return { taskList, hasTaskList: taskList.length > 0 };
      }
    }

    return { taskList: [] as TaskListItem[], hasTaskList: false };
  }, [events]);
}
