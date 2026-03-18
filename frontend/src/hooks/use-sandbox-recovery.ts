import React from "react";
import { useTranslation } from "react-i18next";
import { useUnifiedResumeConversationSandbox } from "./mutation/use-unified-start-conversation";
import { useUserProviders } from "./use-user-providers";
import { useVisibilityChange } from "./use-visibility-change";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import type { ConversationStatus } from "#/types/conversation-status";
import type { Conversation } from "#/api/open-hands.types";

interface UseSandboxRecoveryOptions {
  conversationId: string | undefined;
  conversationStatus: ConversationStatus | undefined;
  /** Function to refetch the conversation data - used to get fresh status on tab focus */
  refetchConversation?: () => Promise<{
    data: Conversation | null | undefined;
  }>;
  onSuccess?: () => void;
  onError?: (error: Error) => void;
}

/**
 * Hook that handles sandbox recovery based on user intent.
 *
 * Recovery triggers:
 * - Page refresh: Resumes the sandbox on initial load if it was paused/stopped
 * - Tab gains focus: Resumes the sandbox if it was paused/stopped
 *
 * What does NOT trigger recovery:
 * - WebSocket disconnect: Does NOT automatically resume the sandbox
 *   (The server pauses sandboxes after 20 minutes of inactivity,
 *    and sandboxes should only be resumed when the user explicitly shows intent)
 *
 * @param options.conversationId - The conversation ID to recover
 * @param options.conversationStatus - The current conversation status
 * @param options.refetchConversation - Function to refetch conversation data on tab focus
 * @param options.onSuccess - Callback when recovery succeeds
 * @param options.onError - Callback when recovery fails
 * @returns isResuming - Whether a recovery is in progress
 */
export function useSandboxRecovery({
  conversationId,
  conversationStatus,
  refetchConversation,
  onSuccess,
  onError,
}: UseSandboxRecoveryOptions) {
  const { t } = useTranslation();
  const { providers } = useUserProviders();
  const { mutate: resumeSandbox, isPending: isResuming } =
    useUnifiedResumeConversationSandbox();

  // Track which conversation ID we've already processed for initial load recovery
  const processedConversationIdRef = React.useRef<string | null>(null);

  const attemptRecovery = React.useCallback(
    (statusOverride?: ConversationStatus) => {
      const status = statusOverride ?? conversationStatus;
      /**
       * Only recover if sandbox is paused (status === STOPPED) and not already resuming
       *
       * Note: ConversationStatus uses different terminology than SandboxStatus:
       *   - SandboxStatus.PAUSED  → ConversationStatus.STOPPED : the runtime is not running but may be restarted
       *   - SandboxStatus.MISSING → ConversationStatus.ARCHIVED : the runtime is not running and will not restart due to deleted files.
       */
      if (!conversationId || status !== "STOPPED" || isResuming) {
        return;
      }

      resumeSandbox(
        { conversationId, providers },
        {
          onSuccess: () => {
            onSuccess?.();
          },
          onError: (error) => {
            displayErrorToast(
              t(I18nKey.CONVERSATION$FAILED_TO_START_WITH_ERROR, {
                error: error.message,
              }),
            );
            onError?.(error);
          },
        },
      );
    },
    [
      conversationId,
      conversationStatus,
      isResuming,
      providers,
      resumeSandbox,
      onSuccess,
      onError,
      t,
    ],
  );

  // Handle page refresh (initial load) and conversation navigation
  React.useEffect(() => {
    if (!conversationId || !conversationStatus) return;

    // Only attempt recovery once per conversation (handles both initial load and navigation)
    if (processedConversationIdRef.current === conversationId) return;

    processedConversationIdRef.current = conversationId;

    if (conversationStatus === "STOPPED") {
      attemptRecovery();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId, conversationStatus]);

  const handleVisible = React.useCallback(async () => {
    // Skip if no conversation or refetch function
    if (!conversationId || !refetchConversation) return;

    try {
      // Refetch to get fresh status - cached status may be stale if sandbox was paused while tab was inactive
      const { data } = await refetchConversation();
      attemptRecovery(data?.status);
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error(
        "Failed to refetch conversation on visibility change:",
        error,
      );
    }
  }, [conversationId, refetchConversation, isResuming, attemptRecovery]);

  // Handle tab focus (visibility change) - refetch conversation status and resume if needed
  useVisibilityChange({
    enabled: !!conversationId,
    onVisible: handleVisible,
  });

  return { isResuming };
}
