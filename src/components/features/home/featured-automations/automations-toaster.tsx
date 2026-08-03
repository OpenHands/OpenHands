import { useLocalStorage } from "@uidotdev/usehooks";
import { Clock } from "lucide-react";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { useNavigation } from "#/context/navigation-context";
import { useHomeAutomations } from "#/hooks/query/use-home-automations";
import { I18nKey } from "#/i18n/declaration";
import CloseIcon from "#/icons/close.svg?react";
import { Typography } from "#/ui/typography";

export const HOME_AUTOMATIONS_TOASTER_DISMISSED_KEY =
  "oh:home-automations-toaster-dismissed";

/**
 * Dismissible toaster-style CTA at the top of the home page that points users
 * at the automations page. Self-gates on automation-service health and a
 * localStorage dismiss flag.
 */
export function AutomationsToaster() {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();
  const { isBackendHealthy, isHealthLoading } = useHomeAutomations();
  const [isDismissed, setIsDismissed] = useLocalStorage(
    HOME_AUTOMATIONS_TOASTER_DISMISSED_KEY,
    false,
  );

  if (isHealthLoading || !isBackendHealthy || isDismissed) {
    return null;
  }

  return (
    <div
      data-testid="home-automations-toaster"
      role="status"
      className="flex w-full flex-col gap-3 rounded-xl border border-[var(--oh-border)] bg-[var(--oh-surface-raised)] px-4 py-3 text-[var(--oh-foreground)] shadow-sm sm:flex-row sm:items-center sm:justify-between sm:py-3.5"
    >
      <div className="flex min-w-0 items-start gap-3">
        <Clock
          className="mt-0.5 size-4 shrink-0 text-white"
          strokeWidth={2}
          aria-hidden="true"
        />
        <div className="min-w-0 space-y-0.5">
          <Typography.Text className="block text-sm font-medium">
            {t(I18nKey.FEATURED_AUTOMATIONS$TOASTER_TITLE)}
          </Typography.Text>
          <Typography.Text className="block text-sm text-[var(--oh-text-secondary)]">
            {t(I18nKey.FEATURED_AUTOMATIONS$TOASTER_MESSAGE)}
          </Typography.Text>
        </div>
      </div>

      <div className="flex shrink-0 items-center gap-2 self-end sm:self-auto">
        <BrandButton
          testId="home-automations-toaster-start"
          type="button"
          variant="primary"
          className="w-fit whitespace-nowrap"
          onClick={() => navigate("/automations")}
        >
          {t(I18nKey.FEATURED_AUTOMATIONS$TOASTER_START)}
        </BrandButton>
        <button
          type="button"
          data-testid="home-automations-toaster-dismiss"
          aria-label={t(I18nKey.FEATURED_AUTOMATIONS$TOASTER_DISMISS)}
          className="inline-flex size-8 items-center justify-center rounded-md text-[var(--oh-text-secondary)] transition-colors hover:bg-[var(--oh-interactive-hover)] hover:text-[var(--oh-foreground)] focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--oh-focus)]"
          onClick={() => setIsDismissed(true)}
        >
          <CloseIcon width={16} height={16} aria-hidden="true" />
        </button>
      </div>
    </div>
  );
}
