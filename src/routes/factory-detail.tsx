import { useMemo, useState, type ReactNode } from "react";
import { Pencil } from "lucide-react";
import { useTranslation } from "react-i18next";
import { useParams } from "react-router";
import AutomationsIcon from "#/icons/automations.svg?react";
import { I18nKey } from "#/i18n/declaration";
import { AutomationsDashboard } from "#/components/features/automations/dashboard/automations-dashboard";
import { AutomationCardSkeleton } from "#/components/features/automations/automation-card-skeleton";
import { ErrorState } from "#/components/features/automations/error-state";
import {
  ManifestSubpageLayout,
  type SubPageNavItem,
} from "#/components/features/manifest/manifest-subpage-layout";
import { KebabMenu } from "#/components/features/automations/kebab-menu";
import { FactoryNameModal } from "#/components/features/factories/factory-name-modal";
import { BackNavButton } from "#/components/shared/buttons/back-nav-button";
import { FACTORIES_PATH, factoryPath } from "#/constants/factories";
import {
  useFactory,
  useFactoriesSupported,
  useRenameFactory,
} from "#/hooks/query/use-factories";
import { useAutomationPermissions } from "#/hooks/use-automation-permissions";
import { getApiErrorMessage } from "#/utils/api-error-message";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
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
  const { canManage, isLoading: isPermissionsLoading } =
    useAutomationPermissions();
  const renameFactory = useRenameFactory();
  const [isRenameOpen, setIsRenameOpen] = useState(false);
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
        Icon: AutomationsIcon,
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
      <div className="flex flex-wrap items-center justify-between gap-2">
        {backLink}
        {canManage && !isPermissionsLoading && (
          <div data-testid="factory-actions">
            <KebabMenu
              items={[
                {
                  label: t(I18nKey.FACTORIES$RENAME),
                  icon: <Pencil className="size-4" aria-hidden />,
                  onClick: () => setIsRenameOpen(true),
                },
              ]}
            />
          </div>
        )}
      </div>
      <FactoryNameModal
        isOpen={isRenameOpen}
        title={t(I18nKey.FACTORIES$RENAME_TITLE)}
        submitLabel={t(I18nKey.FACTORIES$RENAME)}
        initialName={factory.name}
        isPending={renameFactory.isPending}
        onSubmit={(name) =>
          renameFactory.mutate(
            { id: factory.id, name },
            {
              onSuccess: () => setIsRenameOpen(false),
              onError: (error) =>
                displayErrorToast(
                  getApiErrorMessage(error, t(I18nKey.FACTORIES$RENAME_ERROR)),
                ),
            },
          )
        }
        onCancel={() => setIsRenameOpen(false)}
      />
      <AutomationsDashboard factoryId={factory.id} embedded />
    </ManifestSubpageLayout>
  );
}
