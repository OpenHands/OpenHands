import React from "react";
import { useTranslation } from "react-i18next";
import { ExtensionsNavigation } from "#/components/features/skills/extensions-navigation";
import { AddCanvasExtensionModal } from "#/components/features/canvas-extensions/add-canvas-extension-modal";
import { CanvasExtensionCard } from "#/components/features/canvas-extensions/canvas-extension-card";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import {
  CanvasExtensionsUnsupportedError,
  isCanvasExtensionsUnsupportedError,
} from "#/api/canvas-extensions-service";
import { isNoBackend } from "#/api/backend-registry/active-store";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { useCanvasExtensions } from "#/hooks/query/use-canvas-extensions";
import {
  useSetCanvasExtensionEnabled,
  useUninstallCanvasExtension,
} from "#/hooks/mutation/use-manage-canvas-extensions";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { settingsLikeMainScrollClassName } from "#/utils/settings-like-page-layout-classes";
import {
  extensionModuleCardGridClassName,
  extensionModuleCardGridContainerClassName,
  extensionModuleEmptyStateClassName,
} from "#/utils/extension-module-card-classes";
import type { InstalledCanvasExtensionInfo } from "#/types/canvas-extension";

type PendingAction =
  | { type: "enable"; extension: InstalledCanvasExtensionInfo }
  | { type: "uninstall"; extension: InstalledCanvasExtensionInfo };

export default function CanvasExtensionsScreen() {
  const { t } = useTranslation("openhands");
  const { backend } = useActiveBackend();
  const [showAddModal, setShowAddModal] = React.useState(false);
  const [pendingAction, setPendingAction] =
    React.useState<PendingAction | null>(null);
  const query = useCanvasExtensions();
  const setEnabled = useSetCanvasExtensionEnabled();
  const uninstall = useUninstallCanvasExtension();

  const backendCanSupportExtensions =
    !isNoBackend(backend) && backend.kind === "local";
  const unsupported =
    !backendCanSupportExtensions ||
    isCanvasExtensionsUnsupportedError(query.error);
  const isBusy = setEnabled.isPending || uninstall.isPending;

  const confirmAction = () => {
    if (!pendingAction) return;
    if (pendingAction.type === "enable") {
      setEnabled.mutate(
        { name: pendingAction.extension.name, enabled: true },
        { onSuccess: () => setPendingAction(null) },
      );
    } else {
      uninstall.mutate(pendingAction.extension.name, {
        onSuccess: () => setPendingAction(null),
      });
    }
  };

  const pendingDisplayName = pendingAction
    ? pendingAction.extension.manifest?.display_name ||
      pendingAction.extension.name
    : "";
  const confirmationText =
    pendingAction?.type === "enable"
      ? t(I18nKey.SETTINGS$CANVAS_EXTENSIONS_ENABLE_CONFIRM, {
          name: pendingDisplayName,
        })
      : pendingAction?.type === "uninstall"
        ? t(I18nKey.SETTINGS$CANVAS_EXTENSIONS_UNINSTALL_CONFIRM, {
            name: pendingDisplayName,
          })
        : "";

  return (
    <div
      data-testid="canvas-extensions-screen"
      className="flex h-full gap-4 md:gap-6 md:pl-8 lg:gap-10 lg:pl-10"
    >
      <ExtensionsNavigation />
      <main className={cn(settingsLikeMainScrollClassName, "h-full")}>
        <div className="mx-auto flex w-full min-w-0 max-w-[800px] flex-col gap-6">
          <div className="flex min-w-0 items-start justify-between gap-4">
            <div className="min-w-0 space-y-1">
              <h2 className="text-xl font-semibold leading-6 text-foreground">
                {t(I18nKey.NAV$EXTENSIONS)}
              </h2>
              <p className="max-w-2xl text-sm text-tertiary-light">
                {t(I18nKey.SETTINGS$CANVAS_EXTENSIONS_PAGE_DESCRIPTION)}
              </p>
            </div>
            <BrandButton
              type="button"
              variant="secondary"
              testId="canvas-extensions-add-button"
              isDisabled={
                !backendCanSupportExtensions || unsupported || query.isLoading
              }
              className="flex-shrink-0 whitespace-nowrap"
              onClick={() => setShowAddModal(true)}
            >
              {t(I18nKey.SETTINGS$CANVAS_EXTENSIONS_ADD_BUTTON)}
            </BrandButton>
          </div>

          <div className="rounded-xl border border-amber-400/30 bg-amber-400/5 px-4 py-3 text-sm text-amber-100">
            {t(I18nKey.SETTINGS$CANVAS_EXTENSIONS_TRUST_NOTICE)}
          </div>

          {unsupported ? (
            <div className={extensionModuleEmptyStateClassName}>
              <h3 className="text-sm font-semibold text-white">
                {t(I18nKey.SETUP$UNAVAILABLE_TITLE)}
              </h3>
              <p className="mt-2 text-sm text-tertiary-light">
                {isCanvasExtensionsUnsupportedError(query.error)
                  ? query.error.message
                  : backend.kind === "cloud"
                    ? new CanvasExtensionsUnsupportedError("cloud-backend")
                        .message
                    : new CanvasExtensionsUnsupportedError("no-backend")
                        .message}
              </p>
            </div>
          ) : query.isLoading ? (
            <div className="flex flex-col gap-4">
              {[1, 2].map((index) => (
                <div
                  key={index}
                  className="h-40 animate-pulse rounded-xl bg-base-secondary"
                />
              ))}
            </div>
          ) : query.isError ? (
            <div className={extensionModuleEmptyStateClassName}>
              <p className="text-sm text-tertiary-light">
                {query.error instanceof Error
                  ? query.error.message
                  : t(I18nKey.ERROR$GENERIC)}
              </p>
              <BrandButton
                type="button"
                variant="secondary"
                className="mt-4"
                onClick={() => void query.refetch()}
              >
                {t(I18nKey.AUTOMATIONS$ERROR_RETRY)}
              </BrandButton>
            </div>
          ) : query.data?.length ? (
            <div className={extensionModuleCardGridContainerClassName}>
              <div className={extensionModuleCardGridClassName}>
                {query.data.map((extension) => (
                  <CanvasExtensionCard
                    key={extension.name}
                    extension={extension}
                    isBusy={isBusy}
                    onToggle={() => {
                      if (extension.enabled) {
                        setEnabled.mutate({
                          name: extension.name,
                          enabled: false,
                        });
                      } else {
                        setPendingAction({ type: "enable", extension });
                      }
                    }}
                    onUninstall={() =>
                      setPendingAction({ type: "uninstall", extension })
                    }
                  />
                ))}
              </div>
            </div>
          ) : (
            <div className={extensionModuleEmptyStateClassName}>
              <p className="text-sm text-tertiary-light">
                {t(I18nKey.SETTINGS$CANVAS_EXTENSIONS_EMPTY)}
              </p>
            </div>
          )}
        </div>
      </main>

      {showAddModal ? (
        <AddCanvasExtensionModal onClose={() => setShowAddModal(false)} />
      ) : null}
      {pendingAction ? (
        <ConfirmationModal
          text={confirmationText}
          onCancel={() => setPendingAction(null)}
          onConfirm={confirmAction}
          isConfirming={isBusy}
        />
      ) : null}
    </div>
  );
}
