import type { AppConversationStartTask } from "#/api/conversation-service/agent-server-conversation-service.types";
import { useEffect, useRef } from "react";
import { useQueries, useQueryClient, type Query } from "@tanstack/react-query";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useConversationLaunchStore } from "#/stores/conversation-launch-store";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import { isRetryableError } from "#/api/with-retry";

export const useStartTasks = (limit = 10) => {
  const { backend, orgId } = useActiveBackend();
  const queryClient = useQueryClient();
  const launches = useConversationLaunchStore((state) => state.launches);
  // Only this tab knows the owning organization of an unready start task.
  const ids = Object.values(launches)
    .filter(
      (launch) =>
        launch.backendId === backend.id &&
        launch.orgId === orgId &&
        launch.targetId?.startsWith("task-"),
    )
    .slice(-limit)
    .map((launch) => launch.targetId!.slice("task-".length));
  const queries = useQueries({
    queries: ids.map((taskId) => ({
      queryKey: ["start-task", taskId],
      queryFn: () => AgentServerConversationService.getStartTask(taskId),
      enabled: backend.kind === "cloud",
      retry: (count: number, error: Error) =>
        count < 3 && isRetryableError(error),
      meta: { disableToast: true },
      refetchInterval: (query: Query<AppConversationStartTask | null>) => {
        const task = query.state.data;
        return query.state.status === "error" ||
          !task ||
          task.status === "READY" ||
          task.status === "ERROR"
          ? false
          : 3000;
      },
    })),
  });
  const readyIds = queries
    .filter((query) => query.data?.status === "READY")
    .map((query) => query.data!.id)
    .join(",");
  const seenReady = useRef(new Set<string>());
  useEffect(() => {
    const ready = readyIds.split(",").filter(Boolean);
    if (ready.some((id) => !seenReady.current.has(id))) {
      void queryClient.invalidateQueries({
        queryKey: ["user", "conversations"],
      });
    }
    seenReady.current = new Set(ready);
  }, [readyIds, queryClient]);
  return {
    unresolvedTaskIds: queries.flatMap((query, index) =>
      !query.data ? [ids[index]] : [],
    ),
    failedTaskIds: queries.flatMap((query, index) =>
      query.isError || query.data === null ? [ids[index]] : [],
    ),
    data: queries.flatMap((query) =>
      query.data && query.data.status !== "READY" ? [query.data] : [],
    ),
    isLoading: queries.some((query) => query.isLoading),
  };
};
