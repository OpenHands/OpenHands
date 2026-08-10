import { useMemo } from "react";
import { MetricsModalHeader } from "./metrics-modal-header";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { CostSection } from "./cost-section";
import { UsageSection } from "./usage-section";
import { ContextWindowSection } from "./context-window-section";
import { EmptyState } from "./empty-state";
import useMetricsStore from "#/stores/metrics-store";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useConversationMetrics } from "#/hooks/query/use-conversation-metrics";

interface MetricsModalProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
}

export function MetricsModal({ isOpen, onOpenChange }: MetricsModalProps) {
  const storeMetrics = useMetricsStore();
  const { data: conversation } = useActiveConversation();

  const conversationId = conversation?.id;
  const conversationUrl = conversation?.conversation_url;
  const sessionApiKey = conversation?.session_api_key;

  const { data: conversationMetrics } = useConversationMetrics(
    conversationId,
    conversationUrl,
    sessionApiKey,
    isOpen,
  );

  const metrics = useMemo(() => {
    if (conversationMetrics) {
      return {
        cost: conversationMetrics.accumulated_cost,
        max_budget_per_task: conversationMetrics.max_budget_per_task,
        usage: conversationMetrics.accumulated_token_usage
          ? {
              prompt_tokens:
                conversationMetrics.accumulated_token_usage.prompt_tokens ?? 0,
              completion_tokens:
                conversationMetrics.accumulated_token_usage.completion_tokens ??
                0,
              cache_read_tokens:
                conversationMetrics.accumulated_token_usage.cache_read_tokens ??
                0,
              cache_write_tokens:
                conversationMetrics.accumulated_token_usage
                  .cache_write_tokens ?? 0,
              context_window:
                conversationMetrics.accumulated_token_usage.context_window ?? 0,
              per_turn_token:
                conversationMetrics.accumulated_token_usage.per_turn_token ?? 0,
            }
          : null,
      };
    }

    return storeMetrics;
  }, [conversationMetrics, storeMetrics]);

  if (!isOpen) return null;

  return (
    <ModalBackdrop onClose={() => onOpenChange(false)}>
      <ModalBody
        testID="metrics-modal"
        className="relative items-start border border-[var(--oh-border)]"
      >
        <MetricsModalHeader onClose={() => onOpenChange(false)} />
        <div className="w-full">
          {(metrics?.cost !== null || metrics?.usage !== null) && (
            <div className="rounded-md border border-[var(--oh-border)] bg-surface-raised p-3">
              <div className="grid gap-3">
                <CostSection
                  cost={metrics?.cost ?? null}
                  maxBudgetPerTask={metrics?.max_budget_per_task ?? null}
                />

                {metrics?.usage !== null && (
                  <>
                    <UsageSection usage={metrics.usage} />
                    <ContextWindowSection
                      perTurnToken={metrics.usage.per_turn_token}
                      contextWindow={metrics.usage.context_window}
                    />
                  </>
                )}
              </div>
            </div>
          )}

          {!metrics?.cost && !metrics?.usage && (
            <div className="rounded-md border border-[var(--oh-border)] bg-surface-raised p-3">
              <EmptyState />
            </div>
          )}
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}
