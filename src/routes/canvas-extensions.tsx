import CanvasExtensionsService, {
  type CanvasAppBackendStatus,
} from "#/api/canvas-extensions-service";
import { ProfileScopeList } from "#/components/features/settings/agent-profiles/profile-scope-list";
import { useSearchSecrets } from "#/hooks/query/use-get-secrets";
import React from "react";
import { useTranslation } from "react-i18next";
import { ExternalLink } from "lucide-react";
import {
  ExtensionsNavigation,
  ExtensionsCompactNavigation,
} from "#/components/features/skills/extensions-navigation";
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
  useRefreshCanvasExtension,
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

const CANVAS_EXTENSIONS_DOCS_URL =
  "https://docs.openhands.dev/openhands/usage/agent-canvas/canvas-extensions";

// @spec WN-001 — Apps trust notice uses theme-aware warning feedback
export default function CanvasExtensionsScreen() {
  const { t } = useTranslation("openhands");
  const { backend } = useActiveBackend();
  const [showAddModal, setShowAddModal] = React.useState(false);
  const [pendingAction, setPendingAction] =
    React.useState<PendingAction | null>(null);
  const query = useCanvasExtensions();
  const [approval, setApproval] = React.useState<CanvasAppBackendStatus | null>(
    null,
  );
  const [selectedSecrets, setSelectedSecrets] = React.useState<string[]>([]);
  const [approvalError, setApprovalError] = React.useState(false);
  const catalog = useSearchSecrets({
    enabled: pendingAction?.type === "enable",
  });
  React.useEffect(() => {
    setPendingAction(null);
  }, [backend.id, backend.connectionRevision]);
  React.useEffect(() => {
    let active = true;
    setApproval(null);
    setSelectedSecrets([]);
    setApprovalError(false);
    if (
      pendingAction?.type === "enable" &&
      pendingAction.extension.manifest?.backend
    ) {
      CanvasExtensionsService.backendStatus(
        pendingAction.extension.name,
        backend,
      )
        .then((status) => {
          if (active) {
            setApproval(status);
            setSelectedSecrets(
              status.secret_grant_revision === status.revision
                ? (status.secret_refs ?? [])
                : [],
            );
          }
        })
        .catch(() => {
          if (active) setApprovalError(true);
        });
    }
    return () => {
      active = false;
    };
  }, [pendingAction, backend]);
  const requiredSecrets =
    pendingAction?.extension.manifest?.backend?.secret_refs ?? [];
  const needsBackend =
    pendingAction?.type === "enable" &&
    !!pendingAction.extension.manifest?.backend;
  const missingSecrets = requiredSecrets.filter(
    (name) => !catalog.data.some((secret) => secret.name === name),
  );
  const consentReady =
    !needsBackend ||
    (!!approval?.revision &&
      !approvalError &&
      !catalog.isError &&
      missingSecrets.length === 0 &&
      requiredSecrets.every((name) => selectedSecrets.includes(name)));

  const setEnabled = useSetCanvasExtensionEnabled();
  React.useEffect(() => {
    const refresh = () => {
      void query.refetch();
    };
    window.addEventListener("canvas-app-revoked", refresh);
    return () => window.removeEventListener("canvas-app-revoked", refresh);
  }, [query.refetch]);
  const uninstall = useUninstallCanvasExtension();
  const refresh = useRefreshCanvasExtension();

  const backendCanSupportExtensions =
    !isNoBackend(backend) && backend.kind === "local";
  const unsupported =
    !backendCanSupportExtensions ||
    isCanvasExtensionsUnsupportedError(query.error);
  const isBusy =
    setEnabled.isPending || uninstall.isPending || refresh.isPending;

  const currentBackend = React.useRef(backend);
  currentBackend.current = backend;
  const confirmAction = () => {
    if (!pendingAction || !consentReady) return;
    if (pendingAction.type === "enable") {
      setEnabled.mutate(
        {
          name: pendingAction.extension.name,
          enabled: true,
          backend,
          consent: approval?.revision
            ? {
                revision: approval.revision,
                secret_refs: selectedSecrets,
              }
            : undefined,
        },
        {
          onSuccess: () => {
            if (currentBackend.current === backend) setPendingAction(null);
          },
        },
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
      ? t(I18nKey.SETTINGS$APPS_ENABLE_CONFIRM)
      : pendingAction?.type === "uninstall"
        ? t(I18nKey.SETTINGS$APPS_UNINSTALL_CONFIRM, {
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
        <div className="mx-auto flex w-full min-w-0 max-w-200 flex-col gap-6">
          <ExtensionsCompactNavigation />
          <div className="flex min-w-0 items-start justify-between gap-4">
            <div className="min-w-0 space-y-1">
              <h2 className="text-xl font-semibold leading-6 text-foreground">
                {t(I18nKey.SETTINGS$APPS_PAGE_TITLE)}
              </h2>
              <p className="max-w-2xl text-sm text-tertiary-light">
                {t(I18nKey.SETTINGS$APPS_PAGE_DESCRIPTION)}
              </p>
              <a
                href={CANVAS_EXTENSIONS_DOCS_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
              >
                {t(I18nKey.SETTINGS$APPS_BUILD_LINK)}
                <ExternalLink className="size-3.5" aria-hidden />
              </a>
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
              {t(I18nKey.SETTINGS$APPS_ADD_BUTTON)}
            </BrandButton>
          </div>

          <div className="rounded-xl border border-warning/30 bg-warning/5 px-4 py-3 text-sm text-feedback-warning">
            {t(I18nKey.SETTINGS$APPS_TRUST_NOTICE)}
          </div>

          <h3 className="text-base font-semibold text-foreground">
            {t(I18nKey.SETTINGS$APPS_INSTALLED)}
          </h3>

          {unsupported ? (
            <div className={extensionModuleEmptyStateClassName}>
              <h3 className="text-sm font-semibold text-contrast">
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
                    key={`${backend.id}:${backend.connectionRevision}:${extension.name}`}
                    extension={extension}
                    isBusy={isBusy}
                    onToggle={() => {
                      if (extension.enabled) {
                        setEnabled.mutate({
                          name: extension.name,
                          enabled: false,
                          backend,
                        });
                      } else {
                        setPendingAction({ type: "enable", extension });
                      }
                    }}
                    onRefresh={() => refresh.mutate(extension)}
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
                {t(I18nKey.SETTINGS$APPS_EMPTY)}
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
          confirmText={
            pendingAction.type === "enable"
              ? t(I18nKey.SETTINGS$APPS_ENABLE_ACTION)
              : undefined
          }
          isConfirming={isBusy}
          confirmDisabled={!consentReady}
        >
          {needsBackend && (
            <>
              <p>{t(I18nKey.SETTINGS$APPS_BACKEND_WARNING)}</p>
              <ProfileScopeList
                testId="app-secret"
                items={requiredSecrets.map((name) => ({
                  name,
                  description: catalog.data.find(
                    (secret) => secret.name === name,
                  )?.description,
                }))}
                selected={selectedSecrets}
                isDisabled={
                  isBusy ||
                  !approval ||
                  catalog.isLoading ||
                  missingSecrets.length > 0
                }
                onToggle={(name, checked) =>
                  setSelectedSecrets((previous) =>
                    checked
                      ? [...previous, name]
                      : previous.filter((entry) => entry !== name),
                  )
                }
              />
              {missingSecrets.length > 0 && (
                <p role="alert">
                  {t(I18nKey.SETTINGS$APPS_BACKEND_SECRETS, {
                    names: missingSecrets.join(", "),
                  })}{" "}
                  —{" "}
                  <a href="/settings/secrets">
                    {t(I18nKey.SETTINGS$APPS_BACKEND_SECRETS)}
                  </a>
                </p>
              )}
              {(approvalError || catalog.isError || setEnabled.isError) && (
                <p role="alert">
                  {setEnabled.error instanceof Error
                    ? setEnabled.error.message
                    : t(I18nKey.SETTINGS$APPS_BACKEND_ERROR)}
                </p>
              )}
            </>
          )}
        </ConfirmationModal>
      ) : null}
    </div>
  );
}
