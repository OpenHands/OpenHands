import { useCallback } from "react";
import { useTranslation } from "react-i18next";
import type { StartGoalRequest } from "@openhands/typescript-client";
import { startGoal } from "#/hooks/mutation/conversation-mutation-utils";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
import { I18nKey } from "#/i18n/declaration";
import { GOAL_COMMAND } from "#/utils/constants";

const GOAL_PREFIX = `${GOAL_COMMAND} `;

/** Optional leading "--max <n>" (or "--max=<n>") flag that caps the audit rounds. */
const MAX_FLAG = /^--max(?:=|\s+)(\d+)\s*/;

/**
 * Intercepts "/goal [--max N] <objective>" submissions and starts a goal loop
 * on the agent server: it pursues the objective, judging completion after each
 * run until done or the cap is reached. Live progress streams back as goal
 * ConversationStateUpdateEvents (rendered by GoalStatusBanner). Everything else
 * falls through to `onSubmit`. Passthrough when `conversationId` is null.
 */
export const useGoalInterceptor = (
  conversationId: string | null | undefined,
  onSubmit: (message: string) => void,
) => {
  const { t } = useTranslation();

  return useCallback(
    (message: string) => {
      const trimmed = message.trim();
      const isGoal =
        trimmed === GOAL_COMMAND || trimmed.startsWith(GOAL_PREFIX);
      if (!conversationId || !isGoal) {
        onSubmit(message);
        return;
      }

      let rest = trimmed.slice(GOAL_COMMAND.length).trim();
      let maxIterations: number | undefined;
      const maxMatch = rest.match(MAX_FLAG);
      if (maxMatch) {
        maxIterations = parseInt(maxMatch[1], 10);
        rest = rest.slice(maxMatch[0].length).trim();
      }

      const objective = rest;
      if (!objective) {
        displayErrorToast(t(I18nKey.GOAL$OBJECTIVE_REQUIRED)); // bare /goal — no objective to pursue
        return;
      }

      const request: StartGoalRequest = { objective };
      if (maxIterations && maxIterations >= 1) {
        request.max_iterations = maxIterations;
      }

      startGoal(conversationId, request).catch((err: unknown) => {
        const fallback = t(I18nKey.GOAL$START_FAILED);
        const messageText =
          err instanceof Error && err.message ? err.message : fallback;
        displayErrorToast(messageText);
      });
    },
    [conversationId, onSubmit, t],
  );
};
