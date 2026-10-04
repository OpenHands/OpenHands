import React from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useAgentState } from "#/hooks/use-agent-state";
import { useCondenseConversation } from "#/hooks/mutation/use-condense-conversation";
import {
  type ContextCompactionResult,
  useAwaitContextCompaction,
} from "#/hooks/use-await-context-compaction";
import { AgentState } from "#/types/agent-state";
import useMetricsStore from "#/stores/metrics-store";
import { useEventStore } from "#/stores/use-event-store";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { formatCompactTokenCount } from "#/utils/format-token-count";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

/**
 * Shared compact/condense action used by the Usage panel CTA and the
 * composer context-window popover.
 */
export function useCompactContextAction(perTurnToken: number = 0) {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const { data: conversation } = useActiveConversation();
  const { curAgentState } = useAgentState();
  const { mutate: condense, isPending } = useCondenseConversation();
  const [beforeToken, setBeforeToken] = React.useState<number | null>(null);
  const baselineEventIdsRef = React.useRef<Set<string | number> | null>(null);

  const isAgentBusy =
    curAgentState === AgentState.RUNNING ||
    curAgentState === AgentState.LOADING;
  const isCompacting = isPending || beforeToken !== null;
  const isDisabled = !conversation?.id || isAgentBusy || isCompacting;
  const description = t(I18nKey.CONVERSATION$COMPACT_CONTEXT_DESCRIPTION);

  const handleCompactionComplete = React.useEffectEvent(
    (result: ContextCompactionResult) => {
      setBeforeToken(null);
      baselineEventIdsRef.current = null;
      if (conversation?.id) {
        queryClient.invalidateQueries({
          queryKey: ["conversation-metrics", conversation.id],
        });
      }

      if (result.outcome === "timeout") {
        displayErrorToast(t(I18nKey.CONVERSATION$COMPACT_CONTEXT_FAILED));
        return;
      }

      if (result.savedToken > 0) {
        displaySuccessToast(
          t(I18nKey.CONVERSATION$COMPACT_CONTEXT_COMPLETE, {
            saved: formatCompactTokenCount(result.savedToken),
            before: formatCompactTokenCount(result.beforeToken),
            after: formatCompactTokenCount(result.afterToken),
          }),
        );
        return;
      }

      displaySuccessToast(
        t(I18nKey.CONVERSATION$COMPACT_CONTEXT_COMPLETE_NO_CHANGE),
      );
    },
  );

  useAwaitContextCompaction({
    beforeToken,
    baselineEventIds: beforeToken !== null ? baselineEventIdsRef.current : null,
    onComplete: handleCompactionComplete,
  });

  const handleCompact = () => {
    if (!conversation?.id || isCompacting) return;

    const snapshot =
      useMetricsStore.getState().usage?.per_turn_token ?? perTurnToken;
    // The condense POST can return only after the server already emitted its
    // Condensation event, so the await baseline must be captured *before* the
    // request fires, not when the post-ack effect starts.
    baselineEventIdsRef.current = new Set(useEventStore.getState().eventIds);

    condense(
      {
        conversationId: conversation.id,
        conversationUrl: conversation.conversation_url,
        sessionApiKey: conversation.session_api_key,
      },
      {
        onSuccess: () => {
          setBeforeToken(snapshot);
          displaySuccessToast(t(I18nKey.CONVERSATION$COMPACT_CONTEXT_STARTED));
        },
        onError: (error) => {
          setBeforeToken(null);
          baselineEventIdsRef.current = null;
          displayErrorToast(
            retrieveAxiosErrorMessage(error) ||
              t(I18nKey.CONVERSATION$COMPACT_CONTEXT_FAILED),
          );
        },
      },
    );
  };

  return {
    handleCompact,
    isCompacting,
    isDisabled,
    description,
  };
}
