import { useMemo } from "react";
import useMetricsStore from "#/stores/metrics-store";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useConversationMetrics } from "#/hooks/query/use-conversation-metrics";

export interface ContextWindowUsage {
  perTurnToken: number;
  contextWindow: number;
}

function toContextWindowUsage(
  usage:
    | {
        per_turn_token?: number | null;
        context_window?: number | null;
      }
    | null
    | undefined,
): ContextWindowUsage | null {
  const contextWindow = usage?.context_window ?? 0;
  if (contextWindow <= 0) {
    return null;
  }

  return {
    perTurnToken: usage?.per_turn_token ?? 0,
    contextWindow,
  };
}

export function useContextWindowUsage(): ContextWindowUsage | null {
  const storeUsage = useMetricsStore((state) => state.usage);
  const { data: conversation } = useActiveConversation();

  const { data: conversationMetrics } = useConversationMetrics(
    conversation?.id,
    conversation?.conversation_url,
    conversation?.session_api_key,
    Boolean(conversation?.id),
  );

  return useMemo(() => {
    const fromStore = toContextWindowUsage(storeUsage);
    if (fromStore) {
      return fromStore;
    }

    return toContextWindowUsage(
      conversationMetrics?.accumulated_token_usage ?? null,
    );
  }, [storeUsage, conversationMetrics]);
}
