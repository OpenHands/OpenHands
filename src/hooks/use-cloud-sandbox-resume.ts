import { useEffect, useRef } from "react";

export const CLOUD_SANDBOX_RESUME_RETRY_BASE_DELAY_MS = 3000;
export const CLOUD_SANDBOX_RESUME_RETRY_MAX_DELAY_MS = 30000;
export const CLOUD_SANDBOX_RESUME_MAX_ATTEMPTS = 5;

type ResumeState = {
  key: string;
  attempts: number;
  inFlight: boolean;
  succeeded: boolean;
  nextAttemptAt: number;
  errorReported: boolean;
};

type UseCloudSandboxResumeOptions = {
  enabled: boolean;
  conversationId?: string;
  sandboxId?: string | null;
  sandboxStatus?: string | null;
  dataUpdatedAt: number;
  onResume: (sandboxId: string) => Promise<unknown>;
  onError: () => void;
};

const createResumeState = (key: string): ResumeState => ({
  key,
  attempts: 0,
  inFlight: false,
  succeeded: false,
  nextAttemptAt: 0,
  errorReported: false,
});

/**
 * Resumes a paused cloud sandbox once, retrying transient failures on query
 * updates with a bounded exponential backoff. A successful resume is not sent
 * again until the sandbox leaves PAUSED and later becomes PAUSED again.
 */
export const useCloudSandboxResume = ({
  enabled,
  conversationId,
  sandboxId,
  sandboxStatus,
  dataUpdatedAt,
  onResume,
  onError,
}: UseCloudSandboxResumeOptions) => {
  const stateRef = useRef<ResumeState | null>(null);

  useEffect(() => {
    if (!conversationId || !sandboxId) {
      stateRef.current = null;
      return;
    }

    const stateKey = `${conversationId}:${sandboxId}`;
    if (stateRef.current?.key !== stateKey) {
      stateRef.current = createResumeState(stateKey);
    }

    if (!enabled || sandboxStatus !== "PAUSED") {
      // Replacing the state also makes a late rejection from an old request
      // harmless when the sandbox transitions RUNNING -> PAUSED again.
      stateRef.current = createResumeState(stateKey);
      return;
    }

    const state = stateRef.current;
    if (!state || state.inFlight || state.succeeded) return;
    if (state.attempts >= CLOUD_SANDBOX_RESUME_MAX_ATTEMPTS) return;
    if (Date.now() < state.nextAttemptAt) return;

    state.inFlight = true;
    state.attempts += 1;
    const attemptNumber = state.attempts;

    void onResume(sandboxId)
      .then(() => {
        if (stateRef.current !== state) return;
        state.inFlight = false;
        state.succeeded = true;
        state.nextAttemptAt = 0;
      })
      .catch(() => {
        if (stateRef.current !== state) return;
        state.inFlight = false;
        const retryDelay = Math.min(
          CLOUD_SANDBOX_RESUME_RETRY_MAX_DELAY_MS,
          CLOUD_SANDBOX_RESUME_RETRY_BASE_DELAY_MS * 2 ** (attemptNumber - 1),
        );
        state.nextAttemptAt = Date.now() + retryDelay;
        if (!state.errorReported) {
          state.errorReported = true;
          onError();
        }
      });
  }, [
    enabled,
    conversationId,
    sandboxId,
    sandboxStatus,
    dataUpdatedAt,
    onResume,
    onError,
  ]);
};
