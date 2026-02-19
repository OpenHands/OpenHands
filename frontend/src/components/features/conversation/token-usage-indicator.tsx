import React from "react";
import { useTranslation } from "react-i18next";
import useMetricsStore from "#/stores/metrics-store";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useSandboxMetrics } from "#/hooks/query/use-sandbox-metrics";
import { I18nKey } from "#/i18n/declaration";
import { MetricsModal } from "./metrics-modal/metrics-modal";

function formatTokenCount(count: number): string {
  if (count >= 1_000_000) {
    return `${(count / 1_000_000).toFixed(1)}M`;
  }
  if (count >= 1_000) {
    return `${(count / 1_000).toFixed(1)}k`;
  }
  return count.toString();
}

export function TokenUsageIndicator() {
  const { t } = useTranslation();
  const storeMetrics = useMetricsStore();
  const { data: conversation } = useActiveConversation();

  const [metricsModalOpen, setMetricsModalOpen] = React.useState(false);

  const isV1 = conversation?.conversation_version === "V1";
  const conversationId = conversation?.conversation_id;
  const conversationUrl = conversation?.url;
  const sessionApiKey = conversation?.session_api_key;

  const { data: sandboxMetrics } = useSandboxMetrics(
    conversationId,
    conversationUrl,
    sessionApiKey,
    isV1,
  );

  const metrics = React.useMemo(() => {
    if (isV1 && sandboxMetrics) {
      const usage = sandboxMetrics.accumulated_token_usage;
      return {
        cost: sandboxMetrics.accumulated_cost,
        totalTokens: usage
          ? (usage.prompt_tokens ?? 0) + (usage.completion_tokens ?? 0)
          : 0,
      };
    }

    if (storeMetrics.usage) {
      return {
        cost: storeMetrics.cost,
        totalTokens:
          storeMetrics.usage.prompt_tokens +
          storeMetrics.usage.completion_tokens,
      };
    }

    return null;
  }, [isV1, sandboxMetrics, storeMetrics]);

  if (!metrics || (metrics.cost === null && metrics.totalTokens === 0)) {
    return null;
  }

  return (
    <>
      <button
        type="button"
        onClick={() => setMetricsModalOpen(true)}
        className="flex items-center gap-1.5 px-2 py-0.5 rounded text-xs text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700 transition-colors cursor-pointer"
        title={t(I18nKey.CONVERSATION$METRICS_INFO)}
        data-testid="token-usage-indicator"
      >
        {metrics.cost !== null && <span>${metrics.cost.toFixed(4)}</span>}
        {metrics.cost !== null && metrics.totalTokens > 0 && (
          <span className="text-neutral-600">|</span>
        )}
        {metrics.totalTokens > 0 && (
          <span>
            {formatTokenCount(metrics.totalTokens)}{" "}
            {t(I18nKey.CONVERSATION$TOKENS)}
          </span>
        )}
      </button>

      <MetricsModal
        isOpen={metricsModalOpen}
        onOpenChange={setMetricsModalOpen}
      />
    </>
  );
}
