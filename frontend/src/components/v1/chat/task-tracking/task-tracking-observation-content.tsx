import React from "react";
import { ObservationEvent } from "#/types/v1/core";
import { TaskTrackerObservation } from "#/types/v1/core/base/observation";
import { TaskListSection } from "./task-list-section";

interface TaskTrackingObservationContentProps {
  event: ObservationEvent<TaskTrackerObservation>;
}

export function TaskTrackingObservationContent({
  event,
}: TaskTrackingObservationContentProps): React.ReactNode {
  const { observation } = event;
  const { command, task_list: taskList } = observation;
  const shouldShowTaskList = command === "plan" && taskList.length > 0;

  return (
    <div className="flex flex-col gap-4">
      {/* Task List section - only show for 'plan' command */}
      {shouldShowTaskList && <TaskListSection taskList={taskList} />}
    </div>
  );
}
