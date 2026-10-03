import React from "react";
import { useTranslation } from "react-i18next";
import { Download, Plus, Upload } from "lucide-react";

import { getLockedCloudHost } from "#/api/agent-server-config";
import { setRegisteredBackends } from "#/api/backend-registry/active-store";
import { type Backend } from "#/api/backend-registry/types";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import {
  MODAL_MAX_WIDTH_VIEWPORT,
  modalWidthClassName,
} from "#/components/shared/modals/modal-body";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { useActiveBackendContext } from "#/contexts/active-backend-context";
import { useBackendsHealth } from "#/hooks/query/use-backends-health";
import { useAllCloudOrganizations } from "#/hooks/query/use-cloud-organizations";
import { useCloudCurrentUserId } from "#/hooks/query/use-cloud-current-user-id";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { modalTitleLgClassName } from "#/utils/modal-classes";
import {
  mergePortableBackends,
  parsePortableBackends,
  serializePortableBackends,
} from "#/utils/backend-config-portability";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { BackendFormModal } from "./backend-form-modal";
import { BackendRow } from "./backend-row";
import { DeviceFlowAuth } from "./device-flow-auth";

interface ManageBackendsModalProps {
  onClose: () => void;
  /**
   * Recovery mode is used by the root unavailable-backend gate. There is no
   * app shell behind the modal, so dismiss controls would be misleading.
   */
  recoveryMode?: boolean;
}

interface PendingRemoval {
  id: string;
  name: string;
}

/**
 * Resolve the connected organization label for a backend row, mirroring the
 * per-org labelling used by the backend selector. Cloud API keys are bound to
 * a single org (legacy keys may expose several, joined here); local backends
 * and unresolved/errored lookups yield no label.
 */
function resolveBackendOrgLabel(
  backend: Backend,
  cloudOrgs: ReturnType<typeof useAllCloudOrganizations>,
  currentUserIds: ReturnType<typeof useCloudCurrentUserId>,
  personalWorkspaceLabel: string,
): string | undefined {
  if (backend.kind !== "cloud") return undefined;
  const entry = cloudOrgs[backend.id];
  if (!entry || entry.orgs.length === 0) return undefined;
  const userId = currentUserIds[backend.id]?.userId ?? null;
  return entry.orgs
    .map((org) =>
      !!userId && userId === org.id ? personalWorkspaceLabel : org.name,
    )
    .join(", ");
}

export function ManageBackendsModal({
  onClose,
  recoveryMode = false,
}: ManageBackendsModalProps) {
  const { t } = useTranslation("openhands");
  const { backends, active, removeBackend, setActive, updateBackend } =
    useActiveBackendContext();
  const healthByBackendId = useBackendsHealth(backends, {
    probeDisabledOnce: true,
  });
  const cloudOrgs = useAllCloudOrganizations();
  const currentUserIds = useCloudCurrentUserId();
  const personalWorkspaceLabel = t(I18nKey.BACKEND$PERSONAL_WORKSPACE);
  const lockedCloudHost = getLockedCloudHost();
  const isLockedToCloud = lockedCloudHost !== null;
  // Cookie-auth (OHE-hosted) backends have no API key to refresh — re-auth is
  // the main-app login, so the device-flow reconnect controls don't apply.
  const isCookieAuthBackend = active.backend.authMode === "cookie";
  const lockedCloudBackend =
    isLockedToCloud && !isCookieAuthBackend && active.backend.kind === "cloud"
      ? active.backend
      : null;
  const lockedCloudReconnectHost =
    lockedCloudHost ?? lockedCloudBackend?.host ?? "";
  const modalTitle =
    isLockedToCloud && !isCookieAuthBackend
      ? t(I18nKey.BACKEND$RECONNECT_CLOUD_TITLE)
      : t(I18nKey.BACKEND$MANAGE_TITLE);
  const [pendingRemoval, setPendingRemoval] =
    React.useState<PendingRemoval | null>(null);
  const [editingBackend, setEditingBackend] = React.useState<Backend | null>(
    null,
  );
  const [showAddForm, setShowAddForm] = React.useState(false);
  const importInputRef = React.useRef<HTMLInputElement>(null);

  const handleExportBackends = React.useCallback(() => {
    const blob = new Blob([serializePortableBackends(backends)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "openhands-backends.json";
    anchor.click();
    URL.revokeObjectURL(url);
  }, [backends]);

  const handleImportBackends = React.useCallback(
    async (file: File) => {
      try {
        const imported = parsePortableBackends(await file.text());
        setRegisteredBackends(mergePortableBackends(backends, imported));
        displaySuccessToast("Backend configuration imported.");
      } catch (error) {
        displayErrorToast(
          error instanceof Error
            ? error.message
            : "Unable to import backend configuration.",
        );
      } finally {
        if (importInputRef.current) importInputRef.current.value = "";
      }
    },
    [backends],
  );

  const handleConfirmRemoval = () => {
    if (!pendingRemoval) return;
    removeBackend(pendingRemoval.id);
    setPendingRemoval(null);
  };

  const handleSelectBackend = React.useCallback(
    (backend: Backend) => {
      if (active.backend.id !== backend.id || active.orgId !== null) {
        setActive(backend.id);
      }
      onClose();
    },
    [active.backend.id, active.orgId, onClose, setActive],
  );

  const handleCloudLogin = React.useCallback(
    (backend: Backend, apiKey: string) => {
      updateBackend(backend.id, { apiKey });
    },
    [updateBackend],
  );

  const handleLockedCloudReconnect = React.useCallback(
    (apiKey: string) => {
      if (!lockedCloudBackend) return;
      updateBackend(lockedCloudBackend.id, { apiKey });
    },
    [lockedCloudBackend, updateBackend],
  );

  return (
    <>
      <ModalBackdrop
        onClose={recoveryMode ? undefined : onClose}
        closeOnEscape={!recoveryMode}
        closeOnBackdropClick={!recoveryMode}
        aria-label={modalTitle}
      >
        <div
          data-testid="manage-backends-modal"
          className={cn(
            "relative flex flex-col bg-surface border border-border rounded-xl",
            modalWidthClassName("lg"),
            MODAL_MAX_WIDTH_VIEWPORT,
            "max-h-[70vh]",
          )}
        >
          {recoveryMode ? null : (
            <ModalCloseButton
              onClose={onClose}
              testId="close-manage-backends-modal"
            />
          )}
          <div className={cn("p-5", !recoveryMode && "pr-12")}>
            <h2 className={modalTitleLgClassName}>{modalTitle}</h2>
          </div>

          <div className="flex min-h-0 flex-1 flex-col px-5">
            <div
              className="flex-1 overflow-auto rounded-md border border-border bg-surface-raised custom-scrollbar-always"
              data-testid="manage-backends-list"
            >
              {backends.length === 0 ? (
                <p className="px-3 py-6 text-center text-sm text-text-secondary">
                  {t(I18nKey.BACKEND$MANAGE_EMPTY)}
                </p>
              ) : (
                <ul className="divide-y divide-border">
                  {backends.map((backend) => (
                    <BackendRow
                      key={backend.id}
                      backend={backend}
                      health={healthByBackendId[backend.id]}
                      orgLabel={resolveBackendOrgLabel(
                        backend,
                        cloudOrgs,
                        currentUserIds,
                        personalWorkspaceLabel,
                      )}
                      onSelect={() => handleSelectBackend(backend)}
                      onEdit={() => setEditingBackend(backend)}
                      onRemove={() =>
                        setPendingRemoval({
                          id: backend.id,
                          name: backend.name,
                        })
                      }
                      onLogin={
                        backend.authMode === "cookie"
                          ? undefined
                          : (apiKey) => handleCloudLogin(backend, apiKey)
                      }
                    />
                  ))}
                </ul>
              )}
            </div>
          </div>

          {!isLockedToCloud && (
            <div className="px-5 pt-4">
              <div className="flex flex-wrap items-center gap-2">
                <BrandButton
                  type="button"
                  variant="secondary"
                  onClick={handleExportBackends}
                  testId="manage-backends-export"
                  startContent={<Download width={14} height={14} />}
                >
                  {/* eslint-disable-next-line i18next/no-literal-string */}
                  Export
                </BrandButton>
                <BrandButton
                  type="button"
                  variant="secondary"
                  onClick={() => importInputRef.current?.click()}
                  testId="manage-backends-import"
                  startContent={<Upload width={14} height={14} />}
                >
                  {/* eslint-disable-next-line i18next/no-literal-string */}
                  Import
                </BrandButton>
                <input
                  ref={importInputRef}
                  data-testid="manage-backends-import-input"
                  type="file"
                  accept="application/json,.json"
                  className="hidden"
                  onChange={(event) => {
                    const file = event.target.files?.[0];
                    if (file) void handleImportBackends(file);
                  }}
                />
              </div>
              {/* eslint-disable-next-line i18next/no-literal-string */}
              <p className="mt-2 text-xs text-[var(--oh-text-secondary)]">
                Exported backend files contain session API keys. Store and share
                them like credentials.
              </p>
            </div>
          )}

          <div className="flex justify-end gap-2 p-5">
            {isLockedToCloud ? (
              lockedCloudBackend ? (
                <DeviceFlowAuth
                  host={lockedCloudReconnectHost}
                  onSuccess={handleLockedCloudReconnect}
                  testIdRoot="manage-backends-reconnect-cloud"
                  idleButtonLabel={t(I18nKey.BACKEND$RECONNECT_CLOUD)}
                  className="w-full sm:w-auto"
                  buttonClassName="w-full sm:w-auto"
                  statusDisplay="modal"
                />
              ) : null
            ) : (
              <BrandButton
                type="button"
                variant={recoveryMode ? "primary" : "secondary"}
                onClick={() => setShowAddForm(true)}
                testId="manage-backends-add"
                startContent={<Plus width={14} height={14} />}
              >
                {t(I18nKey.BACKEND$ADD)}
              </BrandButton>
            )}
            {recoveryMode ? null : (
              <BrandButton
                type="button"
                variant="primary"
                onClick={onClose}
                testId="manage-backends-done"
              >
                {t(I18nKey.HOME$DONE)}
              </BrandButton>
            )}
          </div>
        </div>
      </ModalBackdrop>

      {showAddForm ? (
        <BackendFormModal
          mode="add"
          source="manage_backends_modal"
          onClose={() => setShowAddForm(false)}
        />
      ) : null}

      {editingBackend ? (
        <BackendFormModal
          mode="edit"
          backend={editingBackend}
          onClose={() => setEditingBackend(null)}
        />
      ) : null}

      {pendingRemoval ? (
        <ConfirmationModal
          text={t(I18nKey.BACKEND$REMOVE_CONFIRMATION, {
            name: pendingRemoval.name,
          })}
          onConfirm={handleConfirmRemoval}
          onCancel={() => setPendingRemoval(null)}
        />
      ) : null}
    </>
  );
}
