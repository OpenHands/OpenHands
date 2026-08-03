import { useTranslation } from "react-i18next";
import { NavigationLink } from "#/components/shared/navigation-link";
import { I18nKey } from "#/i18n/declaration";
import { HOME_RECOMMENDED_AUTOMATION_CARDS } from "./home-recommended-automation-examples";

/**
 * Compact recommended-automation starter cards shown above the home composer.
 * Prototype uses static cards (no catalog fetch).
 */
export function RecommendedAutomationsRail() {
  const { t } = useTranslation("openhands");

  return (
    <section
      data-testid="recommended-automations-rail"
      aria-label={t(I18nKey.FEATURED_AUTOMATIONS$RECOMMENDED_GROUP_LABEL)}
      className="w-full"
    >
      <div
        role="group"
        aria-label={t(I18nKey.FEATURED_AUTOMATIONS$RECOMMENDED_GROUP_LABEL)}
        className="grid grid-cols-2 gap-1.5 sm:grid-cols-4"
      >
        {HOME_RECOMMENDED_AUTOMATION_CARDS.map(
          ({ id, labelKey, Icon, iconColor, href }) => (
            <NavigationLink
              key={id}
              to={href}
              data-testid={`recommended-automation-card-${id}`}
              className="flex min-h-[4.25rem] flex-col justify-between gap-1.5 rounded-lg border border-[var(--oh-border)] bg-[var(--oh-surface-raised)] px-2.5 py-2 text-left transition-colors hover:bg-[var(--oh-interactive-hover)] focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--oh-focus)]"
            >
              <Icon
                aria-hidden="true"
                className="size-3.5 shrink-0"
                color={iconColor}
                strokeWidth={1.75}
              />
              <span className="line-clamp-2 text-[11px] leading-snug text-[var(--oh-foreground)]">
                {t(labelKey)}
              </span>
            </NavigationLink>
          ),
        )}
      </div>
    </section>
  );
}
