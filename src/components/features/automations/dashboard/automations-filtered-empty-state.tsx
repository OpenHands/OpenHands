import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";

interface AutomationsFilteredEmptyStateProps {
  onClear: () => void;
}

/**
 * Automations exist but the active search/filters match none. Chrome copy, so
 * it stays the host's translations rather than manifest copy.
 */
export function AutomationsFilteredEmptyState({
  onClear,
}: AutomationsFilteredEmptyStateProps) {
  const { t } = useTranslation("openhands");
  return (
    <div
      data-testid="automations-filtered-empty"
      className="rounded-xl border border-dashed border-[var(--oh-border)] p-8 text-center"
    >
      <p className="text-sm text-muted">
        {t(I18nKey.AUTOMATIONS$NO_FILTER_MATCHES)}
      </p>
      <button
        type="button"
        data-testid="automations-clear-filters"
        onClick={onClear}
        className="mt-2 text-sm text-white underline-offset-2 hover:underline"
      >
        {t(I18nKey.AUTOMATIONS$CLEAR_FILTERS)}
      </button>
    </div>
  );
}
