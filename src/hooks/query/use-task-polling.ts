import { useEffect, useLayoutEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import AgentServerConversationService from "#/api/conversation-service/agent-server-conversation-service.api";
import type { AppConversationStartTask } from "#/api/conversation-service/agent-server-conversation-service.types";
import {
  getStoredConversationMetadata,
  setStoredConversationMetadata,
  toPluginCoordinates,
} from "#/api/conversation-metadata-store";
import { useNavigation } from "#/context/navigation-context";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";
import {
  consumePendingTaskDraft,
  setConversationState,
} from "#/utils/conversation-local-storage";
import { useOptimisticUserMessageStore } from "#/stores/optimistic-user-message-store";
import { trackCloudConversationReady } from "#/services/cloud-funnel-analytics";
import { flushPendingTaskAttachments } from "#/utils/flush-pending-task-attachments";
import {
  clearPendingTaskMessageLink,
  consumeScheduledPendingTaskMessageReassign,
  linkPendingTaskMessages,
  schedulePendingTaskMessageReassign,
} from "#/utils/pending-task-message-link";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { buildConversationUrl } from "#/api/backend-registry/url-selection";
import { isNoBackend } from "#/api/backend-registry/active-store";

const storeTaskPlugins = (
  task: AppConversationStartTask,
  conversationId: string,
) => {
  const plugins = task.request.plugins?.map(toPluginCoordinates);
  if (!plugins?.length) return;

  const metadata = getStoredConversationMetadata(conversationId);
  setStoredConversationMetadata(conversationId, {
    ...metadata,
    selected_repository:
      metadata?.selected_repository ?? task.request.selected_repository ?? null,
    selected_branch:
      metadata?.selected_branch ?? task.request.selected_branch ?? null,
    git_provider: metadata?.git_provider ?? task.request.git_provider ?? null,
    plugins,
  });
};

/**
 * Read the shared polling state for a V1 conversation start task.
 *
 * This hook:
 * - Detects if the conversationId URL param is a task ID (format: "task-{uuid}")
 * - Polls the V1 start task API every 3 seconds until status is READY or ERROR
 * - Exposes task status and details for UI components to show loading states and errors
 *
 * URL patterns:
 * - /conversations/task-{uuid} → Polls start task, then navigates to /conversations/{conversation-id}
 * - /conversations/{uuid or hex} → No polling (handled by useActiveConversation)
 *
 * The conversation route mounts useTaskPollingController once to own READY
 * side effects; other components can consume this hook without repeating them.
 */
export const useTaskPolling = () => {
  // Optional: the chat input shell renders on the home page too; polling
  // simply no-ops when there's no conversation id yet.
  const { conversationId } = useOptionalConversationId();

  // Check if this is a task ID (format: "task-{uuid}")
  const isTask = !!conversationId && conversationId.startsWith("task-");
  const taskId = isTask ? conversationId!.replace("task-", "") : null;

  // Poll the task if this is a task ID
  const taskQuery = useQuery({
    queryKey: ["start-task", taskId],
    queryFn: async () => {
      if (!taskId) return null;
      return AgentServerConversationService.getStartTask(taskId);
    },
    enabled: !!taskId,
    refetchInterval: (query) => {
      const task = query.state.data;
      if (!task) return false;

      // Stop polling if ready or error
      if (task.status === "READY" || task.status === "ERROR") {
        return false;
      }

      // Poll every 3 seconds while task is in progress
      return 3000;
    },
    retry: false,
  });

  return {
    isTask,
    taskId,
    conversationId: isTask ? null : (conversationId ?? null),
    task: taskQuery.data,
    taskStatus: taskQuery.data?.status,
    taskDetail: taskQuery.data?.detail,
    taskError: taskQuery.error,
    isLoadingTask: taskQuery.isLoading,
    // Repository information from task request
    repositoryInfo: {
      selectedRepository: taskQuery.data?.request?.selected_repository,
      selectedBranch: taskQuery.data?.request?.selected_branch,
      gitProvider: taskQuery.data?.request?.git_provider,
    },
  };
};

/** Own the task lifecycle side effects once at the conversation route. */
export const useTaskPollingController = () => {
  const polling = useTaskPolling();
  const { task, taskId } = polling;
  const { conversationId } = useOptionalConversationId();
  const { navigate } = useNavigation();
  const { backend: activeBackend, orgId: activeOrgId } = useActiveBackend();
  const handledReadyTaskIdRef = useRef<string | null>(null);

  // Reassign optimistic pending messages before paint on the real conversation
  // route. Doing this in the ready handler before navigate leaves a frame where
  // the URL still points at `task-{uuid}` but pending is keyed to the real id.
  useLayoutEffect(() => {
    if (!conversationId) {
      return;
    }

    const pendingReassign =
      consumeScheduledPendingTaskMessageReassign(conversationId);
    if (!pendingReassign) {
      return;
    }

    useOptimisticUserMessageStore
      .getState()
      .reassignPendingMessages(
        pendingReassign.fromConversationId,
        pendingReassign.toConversationId,
      );
    clearPendingTaskMessageLink(pendingReassign.toConversationId);
  }, [conversationId]);

  // Navigate to conversation ID when task is ready
  useEffect(() => {
    const appConversationId = task?.app_conversation_id;
    if (
      !taskId ||
      task?.status !== "READY" ||
      !appConversationId ||
      handledReadyTaskIdRef.current === taskId
    ) {
      return;
    }

    handledReadyTaskIdRef.current = taskId;
    trackCloudConversationReady(taskId, appConversationId);
    storeTaskPlugins(task, appConversationId);

    void (async () => {
      await flushPendingTaskAttachments(taskId, appConversationId);

      const taskConversationId = `task-${taskId}`;
      linkPendingTaskMessages(appConversationId, taskConversationId);
      schedulePendingTaskMessageReassign(taskConversationId, appConversationId);

      const pendingDraft = consumePendingTaskDraft(taskId);
      if (pendingDraft) {
        setConversationState(appConversationId, {
          draftMessage: pendingDraft,
        });
      }

      navigate(
        buildConversationUrl(
          appConversationId,
          isNoBackend(activeBackend)
            ? null
            : {
                backendId: activeBackend.id,
                orgId: activeOrgId,
              },
        ),
        { replace: true },
      );
    })();
  }, [activeBackend.id, activeOrgId, task, taskId, navigate]);

  return polling;
};
