import { useState } from "react";
import { ChevronRight, Plus } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { AutomationCardSkeleton } from "#/components/features/automations/automation-card-skeleton";
import { ErrorState } from "#/components/features/automations/error-state";
import { StatusBadge } from "#/components/features/automations/status-badge";
import {
  automationActivityListClassName,
  automationActivityRowClassName,
} from "#/components/features/automations/automation-view-mode";
import { BrandButton } from "#/components/features/settings/brand-button";
import { FactoryNameModal } from "#/components/features/factories/factory-name-modal";
import { NavigationLink } from "#/components/shared/navigation-link";
import { useNavigation } from "#/context/navigation-context";
import { factoryPath } from "#/constants/factories";
import {
  useCreateFactory,
  useFactories,
  useFactoriesSupported,
} from "#/hooks/query/use-factories";
import { useAutomationPermissions } from "#/hooks/use-automation-permissions";
import { getApiErrorMessage } from "#/utils/api-error-message";
import { displayErrorToast } from "#/utils/custom-toast-handlers";
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
  const { navigate } = useNavigation();
  const { canManage, isLoading: isPermissionsLoading } =
    useAutomationPermissions();
  const createFactory = useCreateFactory();
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const { isLoading: isCapabilitiesLoading, isSupported } =
    useFactoriesSupported();

  const handleCreate = (name: string) => {
    createFactory.mutate(name, {
      onSuccess: (factory) => {
        setIsCreateOpen(false);
        navigate?.(factoryPath(factory.id));
      },
      onError: (error) =>
        displayErrorToast(
          getApiErrorMessage(error, t(I18nKey.FACTORIES$CREATE_ERROR)),
        ),
    });
  };
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
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <h1 className="text-xl font-semibold text-content">
              {t(I18nKey.FACTORIES$NAV_LABEL)}
            </h1>
            <p className="mt-1 text-sm text-muted">
              {t(I18nKey.FACTORIES$LIST_SUBTITLE)}
            </p>
          </div>
          {isSupported && canManage && !isPermissionsLoading && (
            <BrandButton
              type="button"
              variant="primary"
              testId="factories-new"
              className="whitespace-nowrap"
              onClick={() => setIsCreateOpen(true)}
              startContent={<Plus className="size-4" aria-hidden />}
            >
              {t(I18nKey.FACTORIES$NEW)}
            </BrandButton>
          )}
        </div>
        <FactoryNameModal
          isOpen={isCreateOpen}
          title={t(I18nKey.FACTORIES$NEW)}
          submitLabel={t(I18nKey.FACTORIES$CREATE)}
          isPending={createFactory.isPending}
          onSubmit={handleCreate}
          onCancel={() => setIsCreateOpen(false)}
        />
        <div className="mt-6">{renderBody()}</div>
      </div>
    </div>
  );
}
