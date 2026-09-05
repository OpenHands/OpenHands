import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { Button } from "#/components/shared/buttons/button";
import { Tooltip } from "#/components/shared/tooltip/tooltip";

/**
 * Enhanced AutomationsFilteredEmptyState with support agent entry point.
 * Provides a clickable entry for users to get help creating automations.
 */
export function AutomationsFilteredEmptyState({
  onClear,
}: {
  onClear: () => void;
}) {
  const { t } = useTranslation("openhands");

  return (
    <div
      data-testid="automations-filtered-empty"
      className="rounded-xl border border-dashed border-[var(--oh-border)] p-8 text-center"
    >
      <p className="text-sm text-muted mb-4">
        {t(I18nKey.AUTOMATIONS$NO_AUTOMATIONS_YET)}
      </p>

      <Tooltip content={t(I18nKey.AUTOMATIONS$GET_ASSISTANCE_WITH_AUTOMATION)} placement="top">
        <Button
          variant="secondary"
          size="sm"
          data-testid="agent-assistance-entry"
        >
          {t(I18nKey.AUTOMATIONS$AGENT_ASSISTANCE)}
        </Button>
      </Tooltip>

      <Button
        type="button"
        data-testid="automations-clear-filters"
        onClick={onClear}
        className="mt-2 text-sm text-white underline-offset-2 hover:underline"
      >
        {t(I18nKey.AUTOMATIONS$CLEAR_FILTERS)}
      </Button>
    </div>
  );
}

