import { CONVERSATION_QUERY_KEYS } from "./query-keys";
import type { AppConversationStartTask } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useEffect, useRef } from "react";
import {
  useQueries,
  useQueryClient,
  type Query,
  type UseQueryResult,
} from "@tanstack/react-query";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useConversationLaunchStore } from "#/stores/conversation-launch-store";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { isRetryableError } from "#/api/with-retry";

function combineStartTaskQueries(
  queries: UseQueryResult<AppConversationStartTask | null>[],
) {
  return {
    tasks: queries.flatMap((query) => (query.data ? [query.data] : [])),
    unresolvedIndexes: queries.flatMap((query, index) =>
      !query.data ? [index] : [],
    ),
    failedIndexes: queries.flatMap((query, index) =>
      query.isError || query.data === null ? [index] : [],
    ),
    isLoading: queries.some((query) => query.isLoading),
  };
}

export const useStartTasks = (
  limit = 10,
  listedConversationIds: string[] = [],
) => {
  const { backend, orgId } = useActiveBackend();
  const queryClient = useQueryClient();
  const launches = useConversationLaunchStore((state) => state.launches);
  // Only this tab knows the owning organization of an unready start task.
  const ids = Object.values(launches)
    .filter(
      (launch) =>
        (!launch.listed ||
          queryClient.getQueryData<AppConversationStartTask>(
            CONVERSATION_QUERY_KEYS.startTask(
              launch.targetId?.slice(5) ?? null,
            ),
          )?.title_generation?.status === "pending") &&
        launch.backendId === backend.id &&
        launch.orgId === orgId &&
        launch.targetId?.startsWith("task-"),
    )
    .slice(-limit)
    .map((launch) => launch.targetId!.slice("task-".length));
  const { tasks, unresolvedIndexes, failedIndexes, isLoading } = useQueries({
    combine: combineStartTaskQueries,
    queries: ids.map((taskId) => ({
      queryKey: CONVERSATION_QUERY_KEYS.startTask(taskId),
      queryFn: () => AgentServerConversationService.getStartTask(taskId),
      enabled: backend.kind === "cloud",
      retry: (count: number, error: Error) =>
        count < 3 && isRetryableError(error),
      meta: { disableToast: true },
      refetchInterval: (query: Query<AppConversationStartTask | null>) => {
        const task = query.state.data;
        if (
          task?.title_generation?.status === "pending" &&
          Date.now() < Date.parse(task.title_generation.deadline_at) &&
          task.status !== "ERROR"
        )
          return 500;
        return query.state.status === "error" ||
          !task ||
          task.status === "READY" ||
          task.status === "ERROR"
          ? false
          : 3000;
      },
    })),
  });
  const readyIds = tasks
    .filter((task) => task.status === "READY")
    .map((task) => `${task.id}:${task.title_generation?.status ?? "legacy"}`)
    .join(",");
  const seenReady = useRef(new Set<string>());
  useEffect(() => {
    const ready = readyIds.split(",").filter(Boolean);
    if (ready.some((id) => !seenReady.current.has(id))) {
      void queryClient.invalidateQueries({
        queryKey: ["user", "conversations"],
      });
      void queryClient.invalidateQueries({
        queryKey: ["user", "conversation"],
      });
    }
    seenReady.current = new Set(ready);
  }, [readyIds, queryClient]);
  useEffect(() => {
    const { update } = useConversationLaunchStore.getState();
    for (const task of tasks) {
      if (!task?.app_conversation_id) continue;
      const launch = Object.values(launches).find(
        (entry) =>
          entry.backendId === backend.id &&
          entry.orgId === orgId &&
          entry.targetId === `task-${task.id}`,
      );
      if (launch)
        update(launch.id, {
          conversationId: task.app_conversation_id,
          listed: listedConversationIds.includes(task.app_conversation_id),
        });
    }
  }, [tasks, launches, backend.id, orgId, listedConversationIds]);
  return {
    unresolvedTaskIds: unresolvedIndexes.map((index) => ids[index]),
    failedTaskIds: failedIndexes.map((index) => ids[index]),
    data: tasks.filter(
      (task) => !listedConversationIds.includes(task.app_conversation_id ?? ""),
    ),
    isLoading,
  };
};
