import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { UsageBreakdownEntry } from "#/api/conversation-service/agent-server-conversation-service.types";
import { MetricRow } from "../metrics-modal/metric-row";

interface ModelBreakdownProps {
  entries: UsageBreakdownEntry[];
}

/**
 * Per-model cost rows for the usage panel: one row per LLM usage id
 * ("default", "condenser", …) with the model the agent server priced it on.
 * The combined Total Cost row alone cannot answer "which model spent this?".
 */
export function ModelBreakdown({ entries }: ModelBreakdownProps) {
  const { t } = useTranslation("openhands");

  if (entries.length === 0) {
    return null;
  }

  return (
    <div className="border-t border-border-subtle pt-2">
      <span className="text-sm text-muted">
        {t(I18nKey.CONVERSATION$BY_MODEL)}
      </span>
      {entries.map((entry) => (
        <div key={entry.usageId} className="mt-1">
          <MetricRow
            label={entry.modelName || entry.usageId}
            value={`$${entry.accumulatedCost.toFixed(4)}`}
          />
          {entry.usage && (
            <div className="pl-4 text-sm text-muted">
              {t(I18nKey.CONVERSATION$INPUT)}{" "}
              {entry.usage.prompt_tokens.toLocaleString()} ·{" "}
              {t(I18nKey.CONVERSATION$OUTPUT)}{" "}
              {entry.usage.completion_tokens.toLocaleString()}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
