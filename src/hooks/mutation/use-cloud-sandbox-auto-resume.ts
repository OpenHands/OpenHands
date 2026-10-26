import { useEffect, useRef } from "react";
import { useMutation } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { resumeCloudSandbox } from "#/api/cloud/conversation-service.api";
import {
  clearCloudAutoResumeSuppression,
  consumeCloudAutoResumeSuppression,
} from "#/api/cloud/cloud-sandbox-resume-suppression";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import type { BackendKind } from "#/api/backend-registry/types";
import { I18nKey } from "#/i18n/declaration";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

const CLOUD_RESUME_RETRY_DELAY_MS =
  import.meta.env.MODE === "test" ? 10 : 10_000;

type ResumeVariables = {
  resumeKey: string;
  sandboxId: string;
};

type ResumeAttemptState = "pending" | "succeeded";

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
  const activeResumeKeyRef = useRef<string | null>(null);
  const resumeAttemptRef = useRef<{
    key: string;
    state: ResumeAttemptState;
  } | null>(null);

  // TanStack can schedule a retry after the route has moved on; keep the
  // mutation function as the final guard so stale retries become no-ops.
  const { mutate: resumeSandbox } = useMutation({
    mutationFn: async ({ resumeKey, sandboxId }: ResumeVariables) => {
      if (activeResumeKeyRef.current !== resumeKey) return;

      try {
        await resumeCloudSandbox(sandboxId);
      } catch (error) {
        if (activeResumeKeyRef.current === resumeKey) {
          displayErrorToast(t(I18nKey.CONVERSATION$FAILED_TO_START_FROM_TASK));
        }
        throw error;
      }
    },
    retry: () => activeResumeKeyRef.current !== null,
    retryDelay: () => CLOUD_RESUME_RETRY_DELAY_MS,
    onSuccess: (_, variables) => {
      if (activeResumeKeyRef.current === variables.resumeKey) {
        resumeAttemptRef.current = {
          key: variables.resumeKey,
          state: "succeeded",
        };
      }
    },
    meta: { disableToast: true },
  });

  useEffect(
    () => () => {
      clearCloudAutoResumeSuppression(conversationId);
    },
    [conversationId],
  );

  useEffect(() => {
    if (
      backendChanged ||
      !isFetched ||
      !conversation ||
      backendKind !== "cloud" ||
      conversation.sandbox_status !== "PAUSED" ||
      !conversation.sandbox_id
    ) {
      activeResumeKeyRef.current = null;
      return;
    }

    if (consumeCloudAutoResumeSuppression(conversation.id)) {
      activeResumeKeyRef.current = null;
      return;
    }

    const resumeKey = `${conversation.id}:${conversation.sandbox_id}`;
    activeResumeKeyRef.current = resumeKey;

    const currentAttempt = resumeAttemptRef.current;
    if (
      currentAttempt?.key === resumeKey &&
      (currentAttempt.state === "pending" ||
        currentAttempt.state === "succeeded")
    ) {
      return;
    }

    resumeAttemptRef.current = { key: resumeKey, state: "pending" };
    resumeSandbox({ resumeKey, sandboxId: conversation.sandbox_id });

    return () => {
      if (activeResumeKeyRef.current === resumeKey) {
        activeResumeKeyRef.current = null;
      }
      if (
        resumeAttemptRef.current?.key === resumeKey &&
        resumeAttemptRef.current.state !== "succeeded"
      ) {
        resumeAttemptRef.current = null;
      }
    };
  }, [
    backendChanged,
    backendKind,
    conversation?.id,
    conversation?.sandbox_status,
    conversation?.sandbox_id,
    isFetched,
    resumeSandbox,
  ]);
}
