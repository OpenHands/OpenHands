import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { UsageBreakdownEntry } from "#/api/conversation-service/agent-server-conversation-service.types";
import { formatUsageCost } from "#/utils/conversation-metrics";
import { MetricRow } from "../metrics-modal/metric-row";

interface ModelBreakdownProps {
  entries: UsageBreakdownEntry[];
}

/** Per-model rows under Total Cost, so the total can be attributed. */
export function ModelBreakdown({ entries }: ModelBreakdownProps) {
  const { t } = useTranslation("openhands");

  if (entries.length === 0) {
    return null;
  }

  return (
    <div
      className="border-t border-divider pt-2"
      data-testid="usage-model-breakdown"
    >
      <span className="text-sm text-muted">
        {t(I18nKey.CONVERSATION$BY_MODEL)}
      </span>
      {entries.map(({ modelName, accumulatedCost, usage }) => (
        <div key={modelName} className="mt-1">
          <MetricRow
            label={modelName}
            value={formatUsageCost(
              accumulatedCost,
              usage.prompt_tokens + usage.completion_tokens,
            )}
          />
          <div className="pl-4 text-sm text-muted">
            {t(I18nKey.CONVERSATION$INPUT)}{" "}
            {usage.prompt_tokens.toLocaleString()} ·{" "}
            {t(I18nKey.CONVERSATION$OUTPUT)}{" "}
            {usage.completion_tokens.toLocaleString()}
          </div>
        </div>
      ))}
    </div>
  );
}
