import { useEffect, useRef } from "react";
import { useMutation } from "@tanstack/react-query";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useTranslation } from "react-i18next";
import { resumeCloudSandbox } from "#/api/cloud/conversation-service.api";
import {
  clearCloudAutoResumeSuppression,
  consumeCloudAutoResumeSuppression,
} from "#/api/cloud/cloud-sandbox-resume-suppression";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import type { BackendKind } from "#/api/backend-registry/types";
import { I18nKey } from "#/i18n/declaration";
import { retryTransientRequest } from "#/utils/transient-error-retry";
import {
  useOptimisticUserMessageStore,
  type PendingUserMessage,
} from "#/stores/optimistic-user-message-store";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

const CLOUD_RESUME_RETRY_DELAY_MS =
  import.meta.env.MODE === "test" ? 10 : 10_000;

type ResumeAttempt = {
  key: string;
  state: "pending" | "settled";
  message: PendingUserMessage | undefined;
};

type ResumeVariables = {
  attempt: ResumeAttempt;
  sandboxId: string;
};

type UseCloudSandboxAutoResumeOptions = {
  backendChanged: boolean;
  backendKind: BackendKind;
  conversation: AppConversation | null | undefined;
  conversationId: string;
  isFetched: boolean;
};

export function useCloudSandboxAutoResume({
  backendChanged,
  backendKind,
  conversation,
  conversationId,
  isFetched,
}: UseCloudSandboxAutoResumeOptions): void {
  const { t } = useTranslation("openhands");
  const active = useActiveBackend();
  const scope = `${active.backend.id}:${active.orgId}:${conversationId}`;
  const activeResumeKeyRef = useRef<string | null>(null);
  const resumeAttemptRef = useRef<ResumeAttempt | null>(null);
  const sendingMessage = useOptimisticUserMessageStore((state) =>
    state.pendingMessages.find(
      (message) =>
        message.conversationId === conversationId &&
        message.status === "sending",
    ),
  );
  const resumeKey =
    !backendChanged &&
    isFetched &&
    backendKind === "cloud" &&
    conversation?.id === conversationId &&
    conversation.sandbox_status === "PAUSED" &&
    conversation.sandbox_id
      ? `${scope}:${conversation.sandbox_id}`
      : null;

  useEffect(() => {
    activeResumeKeyRef.current = resumeKey;
    return () => {
      activeResumeKeyRef.current = null;
    };
  }, [resumeKey]);

  // TanStack can schedule a retry after the route has moved on; keep the
  // mutation function as the final guard so stale retries become no-ops.
  const { mutate: resumeSandbox } = useMutation({
    mutationFn: async ({ attempt, sandboxId }: ResumeVariables) => {
      if (activeResumeKeyRef.current !== attempt.key) return;

      await resumeCloudSandbox(sandboxId);
    },
    retry: (failureCount, error) =>
      activeResumeKeyRef.current !== null &&
      retryTransientRequest(failureCount, error),
    onError: (_error, { attempt }) => {
      if (activeResumeKeyRef.current !== attempt.key) return;
      const message = t(I18nKey.CONVERSATION$FAILED_TO_START_FROM_TASK);
      const store = useOptimisticUserMessageStore.getState();
      store.pendingMessages
        .filter(
          (pending) =>
            pending.conversationId === conversationId &&
            pending.status === "sending",
        )
        .forEach((pending) =>
          store.markPendingMessageError(pending.id, message),
        );
      displayErrorToast(message);
    },
    retryDelay: () => CLOUD_RESUME_RETRY_DELAY_MS,
    onSettled: (_, __, { attempt }) => {
      if (resumeAttemptRef.current === attempt) {
        resumeAttemptRef.current = { ...attempt, state: "settled" };
      }
    },
    meta: { disableToast: true },
  });

  useEffect(
    () => () => {
      clearCloudAutoResumeSuppression(conversationId);
      resumeAttemptRef.current = null;
    },
    [conversationId, scope],
  );

  useEffect(() => {
    if (!resumeKey || !conversation?.sandbox_id) return;

    if (consumeCloudAutoResumeSuppression(conversationId)) {
      resumeAttemptRef.current = {
        key: resumeKey,
        state: "settled",
        message: sendingMessage,
      };
      return;
    }

    const previous = resumeAttemptRef.current;
    // Repeated pauses need a new send/retry intent, not another idle-tab wake-up.
    if (
      previous?.key === resumeKey &&
      (previous.state === "pending" ||
        !sendingMessage ||
        previous.message === sendingMessage)
    )
      return;

    const attempt: ResumeAttempt = {
      key: resumeKey,
      state: "pending",
      message: sendingMessage,
    };
    resumeAttemptRef.current = attempt;
    resumeSandbox({ attempt, sandboxId: conversation.sandbox_id });
  }, [
    resumeKey,
    conversationId,
    conversation?.sandbox_id,
    sendingMessage,
    resumeSandbox,
  ]);
}
