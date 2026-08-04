import { RefreshCw } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useActiveConversation } from "#/hooks/query/use-active-conversation";
import { useLLMBalance } from "#/hooks/query/use-llm-balance";
import { MetricRow } from "../metrics-modal/metric-row";
import { cn } from "#/utils/utils";

function formatCredits(amount: number): string {
  return `$${amount.toFixed(2)}`;
}

/**
 * Remaining credits for the active LLM provider (OpenRouter). Fetched once
 * per conversation load with a manual refresh icon; hidden entirely when the
 * agent server has no `/api/llm/balance` endpoint (404) or the request
 * fails, so older servers degrade gracefully.
 */
export function ProviderBalanceCard() {
  const { t } = useTranslation("openhands");
  const { data: conversation } = useActiveConversation();
  const {
    data: balance,
    isError,
    isFetching,
    refetch,
  } = useLLMBalance(conversation?.id);

  if (!balance || isError) return null;

  const hasCap = balance.limit !== null && balance.limitRemaining !== null;

  return (
    <div
      data-testid="provider-balance-card"
      className="rounded-md border border-[var(--oh-border)] bg-surface-raised p-3"
    >
      <div className="flex items-center justify-between pb-2">
        <span className="font-semibold">
          {t(I18nKey.CONVERSATION$PROVIDER_BALANCE)}
        </span>
        <div className="flex items-center gap-2">
          <span className="text-xs text-[var(--oh-muted)]">
            {balance.provider}
            {balance.isFreeTier
              ? ` · ${t(I18nKey.CONVERSATION$FREE_TIER)}`
              : null}
          </span>
          <button
            type="button"
            data-testid="provider-balance-refresh"
            aria-label={t(I18nKey.BUTTON$REFRESH)}
            disabled={isFetching}
            onClick={() => refetch()}
            className="cursor-pointer text-[var(--oh-muted)] hover:text-white disabled:cursor-default"
          >
            <RefreshCw
              className={cn("h-3.5 w-3.5", isFetching && "animate-spin")}
            />
          </button>
        </div>
      </div>
      <div className="grid gap-3">
        {hasCap ? (
          <MetricRow
            label={t(I18nKey.CONVERSATION$CREDITS_REMAINING)}
            value={formatCredits(balance.limitRemaining ?? 0)}
          />
        ) : null}
        <MetricRow
          label={t(I18nKey.CONVERSATION$CREDITS_USED)}
          value={formatCredits(balance.usage)}
          valueClassName={hasCap ? "" : "font-semibold"}
          labelClassName={hasCap ? "text-[var(--oh-muted)]" : ""}
        />
      </div>
    </div>
  );
}
