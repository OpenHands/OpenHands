import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { MetricRow } from "./metric-row";

interface UsageSectionProps {
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    cache_read_tokens: number;
    cache_write_tokens: number;
  };
}

export function UsageSection({ usage }: UsageSectionProps) {
  const { t } = useTranslation("openhands");

  return (
    <>
      <MetricRow
        label={t(I18nKey.CONVERSATION$INPUT)}
        value={usage.prompt_tokens.toLocaleString()}
      />

      <div className="grid grid-cols-2 gap-2 pl-4 text-sm">
        <span className="text-[var(--oh-muted)]">
          {t(I18nKey.CONVERSATION$CACHE_HIT)}
        </span>
        <span className="text-right">
          {usage.cache_read_tokens.toLocaleString()}
        </span>
        <span className="text-[var(--oh-muted)]">
          {t(I18nKey.CONVERSATION$CACHE_WRITE)}
        </span>
        <span className="text-right">
          {usage.cache_write_tokens.toLocaleString()}
        </span>
      </div>

      <MetricRow
        label={t(I18nKey.CONVERSATION$OUTPUT)}
        value={usage.completion_tokens.toLocaleString()}
      />

      <MetricRow
        label={t(I18nKey.CONVERSATION$TOTAL)}
        value={(usage.prompt_tokens + usage.completion_tokens).toLocaleString()}
        labelClassName="font-semibold"
        valueClassName="font-bold"
      />
    </>
  );
}
