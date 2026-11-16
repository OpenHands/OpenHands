import { useMemo, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { useParams } from "react-router";
import { I18nKey } from "#/i18n/declaration";
import { AutomationsDashboard } from "#/components/features/automations/dashboard/automations-dashboard";
import { AutomationCardSkeleton } from "#/components/features/automations/automation-card-skeleton";
import { ErrorState } from "#/components/features/automations/error-state";
import { MANIFEST_ICON_BY_SLUG } from "#/components/features/manifest/manifest-icons";
import {
  ManifestSubpageLayout,
  type SubPageNavItem,
} from "#/components/features/manifest/manifest-subpage-layout";
import { BackNavButton } from "#/components/shared/buttons/back-nav-button";
import { FACTORIES_PATH, factoryPath } from "#/constants/factories";
import { useFactory, useFactoriesSupported } from "#/hooks/query/use-factories";
import { getErrorStatus } from "#/hooks/query/use-settings";
import { hasAutomationInterface } from "#/manifests/automation-interface";

/** Factories are served by the automation service, so they need its interface. */
export const clientLoader = () => {
  if (!hasAutomationInterface()) {
    throw new Response(null, { status: 404, statusText: "Not Found" });
  }
  return null;
};

export default function FactoryDetail() {
  const { t } = useTranslation("openhands");
  const { factoryId = "" } = useParams();
  const { isLoading: isCapabilitiesLoading, isSupported } =
    useFactoriesSupported();
  const {
    data: factory,
    isLoading: isFactoryLoading,
    isError,
    error,
    refetch,
  } = useFactory(factoryId, { enabled: isSupported });

  const navItems = useMemo<SubPageNavItem[]>(
    () => [
      {
        to: factoryPath(factoryId),
        label: t(I18nKey.FACTORIES$AUTOMATIONS_NAV),
        Icon: MANIFEST_ICON_BY_SLUG.bot,
        testId: "factory-navigation-automations",
      },
    ],
    [factoryId, t],
  );

  const backLink = (
    <BackNavButton to={FACTORIES_PATH} testId="factory-back-link">
      {t(I18nKey.FACTORIES$BACK)}
    </BackNavButton>
  );

  const renderPlain = (content: ReactNode) => (
    <div className="min-h-full">
      <div className="p-6 max-w-4xl mx-auto flex flex-col gap-6">
        {backLink}
        {content}
      </div>
    </div>
  );

  if (isCapabilitiesLoading || (isSupported && isFactoryLoading)) {
    return renderPlain(<AutomationCardSkeleton />);
  }
  if (!isSupported) {
    return renderPlain(
      <p data-testid="factories-unsupported" className="text-sm text-muted">
        {t(I18nKey.FACTORIES$UNSUPPORTED)}
      </p>,
    );
  }
  if (isError && getErrorStatus(error) === 404) {
    return renderPlain(
      <p data-testid="factory-not-found" className="text-sm text-muted">
        {t(I18nKey.FACTORIES$NOT_FOUND)}
      </p>,
    );
  }
  if (isError || !factory) {
    return renderPlain(
      <ErrorState
        title={t(I18nKey.FACTORIES$LOAD_ERROR)}
        onRetry={() => void refetch()}
      />,
    );
  }

  return (
    <ManifestSubpageLayout
      heading={factory.name}
      navTestIdBase="factory-navbar"
      items={navItems}
    >
      {backLink}
      <AutomationsDashboard factoryId={factory.id} embedded />
    </ManifestSubpageLayout>
  );
}
