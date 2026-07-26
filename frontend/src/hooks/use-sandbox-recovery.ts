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
  refetchConversation?: () => Promise<{
    data: V1AppConversation | null | undefined;
  }>;
  onSuccess?: () => void;
  onError?: (error: Error) => void;
}

export function useSandboxRecovery({
  conversationId,
  sandboxStatus,
  refetchConversation,
  onSuccess,
  onError,
}: UseSandboxRecoveryOptions) {
  const { t } = useTranslation();
  const { providers } = useUserProviders();
  const [
    credentialBindingActivationFailed,
    setCredentialBindingActivationFailed,
  ] = React.useState(false);
  const activeConversationIdRef = React.useRef(conversationId);
  activeConversationIdRef.current = conversationId;
  const recoveryInFlightRef = React.useRef(false);
  const credentialBindingActivationFailedRef = React.useRef(false);
  const { mutate: resumeSandbox, isPending: isResuming } =
    useUnifiedResumeConversationSandbox();

  const processedConversationIdRef = React.useRef<string | null>(null);

  const attemptRecovery = React.useCallback(
    (statusOverride?: V1SandboxStatus, force = false) => {
      const status = statusOverride ?? sandboxStatus;
      if (
        !conversationId ||
        (!force && status !== "PAUSED") ||
        (force && credentialBindingActivationFailedRef.current) ||
        isResuming ||
        recoveryInFlightRef.current
      ) {
        return;
      }
      const recoveryConversationId = conversationId;
      recoveryInFlightRef.current = true;
      resumeSandbox(
        { conversationId: recoveryConversationId, providers },
        {
          onSuccess: () => {
            if (activeConversationIdRef.current !== recoveryConversationId) {
              return;
            }
            recoveryInFlightRef.current = false;
            credentialBindingActivationFailedRef.current = false;
            setCredentialBindingActivationFailed(false);
            onSuccess?.();
          },
          onError: (error) => {
            if (activeConversationIdRef.current !== recoveryConversationId) {
              return;
            }
            recoveryInFlightRef.current = false;
            if (force) {
              credentialBindingActivationFailedRef.current = true;
              setCredentialBindingActivationFailed(true);
            }
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

  React.useEffect(() => {
    recoveryInFlightRef.current = false;
    credentialBindingActivationFailedRef.current = false;
    setCredentialBindingActivationFailed(false);
  }, [conversationId]);

  React.useEffect(() => {
    if (!conversationId || !sandboxStatus) return;

    if (processedConversationIdRef.current === conversationId) return;

    processedConversationIdRef.current = conversationId;

    if (sandboxStatus === "PAUSED") {
      attemptRecovery();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId, sandboxStatus]);

  const handleVisible = React.useCallback(async () => {
    if (!conversationId || !refetchConversation) return;
    const visibleConversationId = conversationId;

    try {
      const { data } = await refetchConversation();
      if (activeConversationIdRef.current !== visibleConversationId) {
        return;
      }
      const retryCredentialBinding =
        credentialBindingActivationFailedRef.current;
      credentialBindingActivationFailedRef.current = false;
      setCredentialBindingActivationFailed(false);
      attemptRecovery(data?.sandbox_status, retryCredentialBinding);
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error(
        "Failed to refetch conversation on visibility change:",
        error,
      );
    }
  }, [conversationId, refetchConversation, attemptRecovery]);

  useVisibilityChange({
    enabled: !!conversationId,
    onVisible: handleVisible,
  });

  const recoverCredentialBinding = React.useCallback(() => {
    attemptRecovery(undefined, true);
  }, [attemptRecovery]);

  return {
    isResuming,
    credentialBindingActivationFailed,
    recoverCredentialBinding,
  };
}
