import React from "react";
import { useTranslation } from "react-i18next";
import { useUnifiedResumeConversationSandbox } from "./mutation/use-unified-start-conversation";
import { useUserProviders } from "./use-user-providers";
import { useVisibilityChange } from "./use-visibility-change";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import { V1SandboxStatus } from "#/api/sandbox-service/sandbox-service.types";
import { V1AppConversation } from "#/api/conversation-service/v1-conversation-service.types";

interface UseSandboxRecoveryOptions {
  conversationId: string | undefined;
  sandboxStatus: V1SandboxStatus | undefined;
  /** Function to refetch the conversation data - used to get fresh status on tab focus */
  refetchConversation?: () => Promise<{
    data: V1AppConversation | null | undefined;
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
  sandboxStatus,
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
    (statusOverride?: V1SandboxStatus) => {
      const status = statusOverride ?? sandboxStatus;
      /**
       * Only recover if sandbox is paused
       */
      if (!conversationId || status !== "PAUSED" || isResuming) {
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
      sandboxStatus,
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
    if (!conversationId || !sandboxStatus) return;

    // Only attempt recovery once per conversation (handles both initial load and navigation)
    if (processedConversationIdRef.current === conversationId) return;

    processedConversationIdRef.current = conversationId;

    if (sandboxStatus === "PAUSED") {
      attemptRecovery();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId, sandboxStatus]);

  const handleVisible = React.useCallback(async () => {
    // Skip if no conversation or refetch function
    if (!conversationId || !refetchConversation) return;

    try {
      // Refetch to get fresh status - cached status may be stale if sandbox was paused while tab was inactive
      const { data } = await refetchConversation();
      attemptRecovery(data?.sandbox_status);
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
