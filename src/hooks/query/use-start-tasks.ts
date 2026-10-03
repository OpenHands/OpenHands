import React from "react";
import { useQuery } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversationStartTask } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { START_TASKS_QUERY_KEYS } from "#/hooks/query/query-keys";
import { useInFlightStartTasksStore } from "#/stores/in-flight-start-tasks-store";

/** Same cadence `useTaskPolling` uses on the conversation route. */
const START_TASK_POLL_INTERVAL_MS = 3000;

const isInProgress = (task: AppConversationStartTask) =>
  task.status !== "READY" && task.status !== "ERROR";

/**
 * Hook to fetch in-progress conversation start tasks
 *
 * Use case: Show tasks that are provisioning sandboxes, cloning repos, etc.
 * These are conversations that started but haven't reached READY or ERROR status yet.
 *
 * The app-server has no start-task search endpoint — only a batch lookup by id
 * — so the ids come from `useInFlightStartTasksStore`, which
 * `useCreateConversation` fills when a launch returns a task that is still
 * provisioning. Settled tasks (and ids the backend no longer knows) are
 * untracked as soon as they come back, so the list drains itself.
 *
 * Note: Filters out READY and ERROR status tasks client-side since backend doesn't support status filtering.
 *
 * @param limit Maximum number of tasks to return (max 100)
 * @returns Query result with array of in-progress start tasks
 */
export const useStartTasks = (limit = 10) => {
  const { backend } = useActiveBackend();
  const backendId = backend.id;
  const trackedTaskIds = useInFlightStartTasksStore(
    (state) => state.taskIdsByBackendId[backendId],
  );
  const untrackStartTasks = useInFlightStartTasksStore(
    (state) => state.untrackStartTasks,
  );

  const taskIds = React.useMemo(
    () => (trackedTaskIds ?? []).slice(0, limit),
    [trackedTaskIds, limit],
  );

  const query = useQuery({
    queryKey: START_TASKS_QUERY_KEYS.byIds(backendId, taskIds),
    queryFn: () => AgentServerConversationService.getStartTasks(taskIds),
    enabled: taskIds.length > 0,
    // `state.data` holds the raw response (`select` doesn't touch it), so this
    // keeps polling for as long as any fetched task is still provisioning.
    refetchInterval: (currentQuery) =>
      currentQuery.state.data?.some(isInProgress)
        ? START_TASK_POLL_INTERVAL_MS
        : false,
    retry: false,
    select: (tasks) => tasks.filter(isInProgress),
  });

  // Stop tracking every id that came back settled, plus any the backend
  // dropped from the batch response — a failed fetch leaves `data` untouched,
  // so nothing is untracked on a transient error.
  const { data: inProgressTasks } = query;
  const settledTaskIds = React.useMemo(() => {
    if (!inProgressTasks) return [];
    const stillRunning = new Set(inProgressTasks.map((task) => task.id));
    return taskIds.filter((id) => !stillRunning.has(id));
  }, [inProgressTasks, taskIds]);

  React.useEffect(() => {
    if (settledTaskIds.length === 0) return;
    untrackStartTasks(backendId, settledTaskIds);
  }, [backendId, settledTaskIds, untrackStartTasks]);

  return query;
};
