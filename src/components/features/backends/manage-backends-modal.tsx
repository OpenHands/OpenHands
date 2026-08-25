import React from "react";
import { useTranslation } from "react-i18next";
import { Download, Plus, Upload } from "lucide-react";

import { getLockedCloudHost } from "#/api/agent-server-config";
import { type Backend } from "#/api/backend-registry/types";
import {
  parsePortableBackendConfig,
  PORTABLE_BACKEND_CONFIG_FILENAME,
  serializePortableBackendConfig,
} from "#/api/backend-registry/portable-config";
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
import { cn, downloadBlob } from "#/utils/utils";
import { modalTitleLgClassName } from "#/utils/modal-classes";
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
  const {
    backends,
    active,
    importBackends,
    removeBackend,
    setActive,
    updateBackend,
  } = useActiveBackendContext();
  const healthByBackendId = useBackendsHealth(backends, {
    probeDisabledOnce: true,
  });
  const cloudOrgs = useAllCloudOrganizations();
  const currentUserIds = useCloudCurrentUserId();
  const personalWorkspaceLabel = t(I18nKey.BACKEND$PERSONAL_WORKSPACE);
  const lockedCloudHost = getLockedCloudHost();
  const isLockedToCloud = lockedCloudHost !== null;
  const lockedCloudBackend =
    isLockedToCloud && active.backend.kind === "cloud" ? active.backend : null;
  const lockedCloudReconnectHost =
    lockedCloudHost ?? lockedCloudBackend?.host ?? "";
  const modalTitle = isLockedToCloud
    ? t(I18nKey.BACKEND$RECONNECT_CLOUD_TITLE)
    : t(I18nKey.BACKEND$MANAGE_TITLE);
  const [pendingRemoval, setPendingRemoval] =
    React.useState<PendingRemoval | null>(null);
  const [editingBackend, setEditingBackend] = React.useState<Backend | null>(
    null,
  );
  const [showAddForm, setShowAddForm] = React.useState(false);
  const [showExportWarning, setShowExportWarning] = React.useState(false);
  const [importError, setImportError] = React.useState(false);
  const [importSucceeded, setImportSucceeded] = React.useState(false);
  const importInputRef = React.useRef<HTMLInputElement>(null);

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

  const handleExport = React.useCallback(() => {
    const blob = new Blob([serializePortableBackendConfig(backends)], {
      type: "application/json",
    });
    downloadBlob(blob, PORTABLE_BACKEND_CONFIG_FILENAME);
    setShowExportWarning(false);
  }, [backends]);

  const handleImport = React.useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const input = event.currentTarget;
      const file = input.files?.[0];
      input.value = "";
      if (!file) return;

      setImportError(false);
      setImportSucceeded(false);
      try {
        const imported = parsePortableBackendConfig(await file.text());
        importBackends(imported);
        setImportSucceeded(true);
      } catch {
        setImportError(true);
      }
    },
    [importBackends],
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
            "relative flex flex-col bg-[var(--oh-surface)] border border-[var(--oh-border)] rounded-xl",
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
              className="flex-1 overflow-auto rounded-md border border-[var(--oh-border)] bg-surface-raised custom-scrollbar-always"
              data-testid="manage-backends-list"
            >
              {backends.length === 0 ? (
                <p className="px-3 py-6 text-center text-sm text-[var(--oh-text-secondary)]">
                  {t(I18nKey.BACKEND$MANAGE_EMPTY)}
                </p>
              ) : (
                <ul className="divide-y divide-[var(--oh-border)]">
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
                      onLogin={(apiKey) => handleCloudLogin(backend, apiKey)}
                    />
                  ))}
                </ul>
              )}
            </div>
          </div>

          {isLockedToCloud ? null : (
            <div className="flex flex-wrap items-center justify-between gap-2 px-5 pt-3">
              <div aria-live="polite" className="text-xs">
                {importError ? (
                  <p role="alert" className="text-danger">
                    {t(I18nKey.BACKEND$IMPORT_ERROR)}
                  </p>
                ) : null}
                {importSucceeded ? (
                  <p className="text-success">
                    {t(I18nKey.BACKEND$IMPORT_SUCCESS)}
                  </p>
                ) : null}
              </div>
              <div className="flex gap-2">
                <BrandButton
                  type="button"
                  variant="secondary"
                  onClick={() => importInputRef.current?.click()}
                  testId="manage-backends-import"
                  startContent={<Upload width={14} height={14} />}
                >
                  {t(I18nKey.BACKEND$IMPORT)}
                </BrandButton>
                <input
                  ref={importInputRef}
                  type="file"
                  accept="application/json,.json"
                  className="sr-only"
                  data-testid="manage-backends-import-input"
                  aria-label={t(I18nKey.BACKEND$IMPORT)}
                  onChange={handleImport}
                />
                <BrandButton
                  type="button"
                  variant="secondary"
                  onClick={() => setShowExportWarning(true)}
                  testId="manage-backends-export"
                  startContent={<Download width={14} height={14} />}
                >
                  {t(I18nKey.BACKEND$EXPORT)}
                </BrandButton>
              </div>
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

      {showExportWarning ? (
        <ConfirmationModal
          text={t(I18nKey.BACKEND$EXPORT_WARNING)}
          onConfirm={handleExport}
          onCancel={() => setShowExportWarning(false)}
        />
      ) : null}
    </>
  );
}
