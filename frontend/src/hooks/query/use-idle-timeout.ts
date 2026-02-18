import { useQuery } from "@tanstack/react-query";
import { SandboxService } from "#/api/sandbox-service/sandbox-service.api";
import { useBatchAppConversations } from "./use-batch-app-conversations";
import { useActiveConversation } from "./use-active-conversation";

/**
 * Polls the sandbox idle timeout status for the current V1 conversation.
 *
 * Returns `isWarning: true` when the sandbox is within the warning window
 * before being auto-paused due to inactivity. `remainingSeconds` counts
 * down to the auto-pause moment.
 */
export const useIdleTimeout = () => {
  const { data: conversation } = useActiveConversation();
  const isV1 = conversation?.conversation_version === "V1";
  const conversationId = conversation?.conversation_id;

  // Resolve conversation → sandbox_id via V1 app conversation
  const appConversationsQuery = useBatchAppConversations(
    isV1 && conversationId ? [conversationId] : [],
  );
  const sandboxId = appConversationsQuery.data?.[0]?.sandbox_id;

  const idleStatusQuery = useQuery({
    queryKey: ["sandbox-idle-status", sandboxId],
    queryFn: () => SandboxService.getIdleStatus(sandboxId!),
    enabled: !!sandboxId,
    // Poll every 30 seconds; increase to every 10 seconds when in warning state
    refetchInterval: (query) => {
      const status = query.state.data;
      if (!status || status.timeout_seconds === 0) return false; // disabled
      if (status.is_warning) return 10_000; // 10s during warning
      return 30_000; // 30s normally
    },
    // Don't show error toasts for this background poll
    meta: { disableToast: true },
  });

  const status = idleStatusQuery.data;

  return {
    isWarning: status?.is_warning ?? false,
    remainingSeconds: status?.remaining_seconds ?? 0,
    timeoutSeconds: status?.timeout_seconds ?? 0,
    isEnabled: (status?.timeout_seconds ?? 0) > 0,
  };
};
