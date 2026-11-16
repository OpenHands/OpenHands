import { ChevronRight } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { AutomationCardSkeleton } from "#/components/features/automations/automation-card-skeleton";
import { ErrorState } from "#/components/features/automations/error-state";
import { StatusBadge } from "#/components/features/automations/status-badge";
import {
  automationActivityListClassName,
  automationActivityRowClassName,
} from "#/components/features/automations/automation-view-mode";
import { NavigationLink } from "#/components/shared/navigation-link";
import { factoryPath } from "#/constants/factories";
import {
  useFactories,
  useFactoriesSupported,
} from "#/hooks/query/use-factories";
import { hasAutomationInterface } from "#/manifests/automation-interface";

/** Factories are served by the automation service, so they need its interface. */
export const clientLoader = () => {
  if (!hasAutomationInterface()) {
    throw new Response(null, { status: 404, statusText: "Not Found" });
  }
  return null;
};

export default function FactoriesList() {
  const { t } = useTranslation("openhands");
  const { isLoading: isCapabilitiesLoading, isSupported } =
    useFactoriesSupported();
  const {
    data: factories,
    isLoading: isFactoriesLoading,
    isError,
    refetch,
  } = useFactories({ enabled: isSupported });

  const renderBody = () => {
    if (isCapabilitiesLoading || (isSupported && isFactoriesLoading)) {
      return (
        <div className="flex flex-col gap-3">
          <AutomationCardSkeleton />
          <AutomationCardSkeleton />
        </div>
      );
    }
    if (!isSupported) {
      return (
        <p data-testid="factories-unsupported" className="text-sm text-muted">
          {t(I18nKey.FACTORIES$UNSUPPORTED)}
        </p>
      );
    }
    if (isError) {
      return (
        <ErrorState
          title={t(I18nKey.FACTORIES$LOAD_ERROR)}
          onRetry={() => void refetch()}
        />
      );
    }
    if (!factories?.length) {
      return (
        <div data-testid="factories-empty" className="flex flex-col gap-1">
          <p className="text-sm font-medium text-content">
            {t(I18nKey.FACTORIES$NONE)}
          </p>
          <p className="text-sm text-muted">
            {t(I18nKey.FACTORIES$NONE_DESCRIPTION)}
          </p>
        </div>
      );
    }
    return (
      <ul
        data-testid="factories-list"
        className={automationActivityListClassName}
      >
        {factories.map((factory) => (
          <li
            key={factory.id}
            data-testid={`factory-row-${factory.id}`}
            className={automationActivityRowClassName}
          >
            <NavigationLink
              to={factoryPath(factory.id)}
              data-testid={`factory-link-${factory.id}`}
              className="flex min-w-0 flex-1 items-center gap-2 px-3 py-3 focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-focus"
            >
              <span className="min-w-0 truncate text-sm font-medium text-foreground">
                {factory.name}
              </span>
              <StatusBadge count={factory.automation_count} />
              <ChevronRight
                className="ml-auto size-4 shrink-0 text-muted"
                aria-hidden
              />
            </NavigationLink>
          </li>
        ))}
      </ul>
    );
  };

  return (
    <div className="min-h-full">
      <div className="p-6 max-w-4xl mx-auto">
        <h1 className="text-xl font-semibold text-content">
          {t(I18nKey.FACTORIES$NAV_LABEL)}
        </h1>
        <p className="mt-1 text-sm text-muted">
          {t(I18nKey.FACTORIES$LIST_SUBTITLE)}
        </p>
        <div className="mt-6">{renderBody()}</div>
      </div>
    </div>
  );
}
