import React from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useUnifiedResumeConversationSandbox } from "#/hooks/mutation/use-unified-start-conversation";
import { useUserProviders } from "#/hooks/use-user-providers";
import { useErrorMessageStore } from "#/stores/error-message-store";
import { I18nKey } from "#/i18n/declaration";

const MAX_RECOVERY_ATTEMPTS = 3;
const RECOVERY_COOLDOWN_MS = 5000;
const RECOVERY_SETTLED_DELAY_MS = 2000;

/**
 * Hook that handles silent WebSocket recovery by resuming the sandbox
 * when a WebSocket disconnection is detected.
 *
 * @param conversationId - The conversation ID to recover
 * @returns reconnectKey - Key to force provider remount (resets connection state)
 * @returns handleDisconnect - Callback to trigger recovery on WebSocket disconnect
 */
export function useWebSocketRecovery(conversationId: string) {
  // Recovery state (refs to avoid re-renders)
  const recoveryAttemptsRef = React.useRef(0);
  const recoveryInProgressRef = React.useRef(false);
  const lastRecoveryAttemptRef = React.useRef<number | null>(null);

  // Key to force remount of provider after recovery (resets connection state to "CONNECTING")
  const [reconnectKey, setReconnectKey] = React.useState(0);

  const queryClient = useQueryClient();
  const { mutate: resumeConversation } = useUnifiedResumeConversationSandbox();
  const { providers } = useUserProviders();
  const setErrorMessage = useErrorMessageStore(
    (state) => state.setErrorMessage,
  );

  // Reset recovery state when conversation changes
  React.useEffect(() => {
    recoveryAttemptsRef.current = 0;
    recoveryInProgressRef.current = false;
    lastRecoveryAttemptRef.current = null;
  }, [conversationId]);

  // Silent recovery callback - resumes sandbox when WebSocket disconnects
  const handleDisconnect = React.useCallback(() => {
    // Prevent concurrent recovery attempts
    if (recoveryInProgressRef.current) return;

    // Check cooldown
    const now = Date.now();
    if (
      lastRecoveryAttemptRef.current &&
      now - lastRecoveryAttemptRef.current < RECOVERY_COOLDOWN_MS
    ) {
      return;
    }

    // Check max attempts - notify user when recovery is exhausted
    if (recoveryAttemptsRef.current >= MAX_RECOVERY_ATTEMPTS) {
      setErrorMessage(I18nKey.STATUS$CONNECTION_LOST);
      return;
    }

    // Start silent recovery
    recoveryInProgressRef.current = true;
    lastRecoveryAttemptRef.current = now;
    recoveryAttemptsRef.current += 1;

    resumeConversation(
      { conversationId, providers },
      {
        onSuccess: async () => {
          // Invalidate and wait for refetch to complete before remounting
          // This ensures the provider remounts with fresh data (url: null during startup)
          await queryClient.invalidateQueries({
            queryKey: ["user", "conversation", conversationId],
          });

          // Force remount to reset connection state to "CONNECTING"
          setReconnectKey((k) => k + 1);

          // Reset recovery state on success
          recoveryAttemptsRef.current = 0;
          recoveryInProgressRef.current = false;
          lastRecoveryAttemptRef.current = null;
        },
        onError: () => {
          // If this was the last attempt, show error to user
          if (recoveryAttemptsRef.current >= MAX_RECOVERY_ATTEMPTS) {
            setErrorMessage(I18nKey.STATUS$CONNECTION_LOST);
          }
          // recoveryInProgressRef will be reset by onSettled
        },
        onSettled: () => {
          // Allow next attempt after a delay (covers both success and error)
          setTimeout(() => {
            recoveryInProgressRef.current = false;
          }, RECOVERY_SETTLED_DELAY_MS);
        },
      },
    );
  }, [
    conversationId,
    providers,
    resumeConversation,
    queryClient,
    setErrorMessage,
  ]);

  return { reconnectKey, handleDisconnect };
}
