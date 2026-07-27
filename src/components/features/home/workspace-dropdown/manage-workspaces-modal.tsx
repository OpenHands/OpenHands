import React from "react";
import { useTranslation } from "react-i18next";

import { BaseModalTitle } from "#/components/shared/modals/confirmation-modals/base-modal";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import {
  MODAL_MAX_WIDTH_VIEWPORT,
  modalWidthClassName,
} from "#/components/shared/modals/modal-body";
import { ConfirmationModal } from "#/components/shared/modals/confirmation-modal";
import { BrandButton } from "#/components/features/settings/brand-button";
import { I18nKey } from "#/i18n/declaration";
import { LocalWorkspace, LocalWorkspaceParent } from "#/types/workspace";
import { cn } from "#/utils/utils";
import { modalTitleSmClassName } from "#/utils/modal-classes";
import FolderIcon from "#/icons/folder.svg?react";
import CloseIcon from "#/icons/close.svg?react";

interface ManageWorkspacesModalProps {
  isOpen: boolean;
  workspaces: LocalWorkspace[];
  workspaceParents?: LocalWorkspaceParent[];
  onClose: () => void;
  onRemove: (path: string) => void;
  onRemoveParent?: (path: string) => void;
}

type PendingRemoval =
  | { type: "workspace"; path: string; text: string }
  | { type: "parent"; path: string; text: string };

export function ManageWorkspacesModal({
  isOpen,
  workspaces,
  workspaceParents = [],
  onClose,
  onRemove,
  onRemoveParent,
}: ManageWorkspacesModalProps) {
  const { t } = useTranslation("openhands");
  const [pendingRemoval, setPendingRemoval] =
    React.useState<PendingRemoval | null>(null);

  if (!isOpen) return null;

  // Workspaces from a parent are read-only here; users remove the parent.
  const staticWorkspaces = workspaces.filter((w) => !w.parentPath);
  const dynamicWorkspacesByParent = new Map<string, LocalWorkspace[]>();
  workspaces.forEach((w) => {
    if (!w.parentPath) return;
    const list = dynamicWorkspacesByParent.get(w.parentPath) ?? [];
    list.push(w);
    dynamicWorkspacesByParent.set(w.parentPath, list);
  });

  const hasContent = staticWorkspaces.length > 0 || workspaceParents.length > 0;

  const handleConfirmRemoval = () => {
    if (!pendingRemoval) return;

    if (pendingRemoval.type === "workspace") {
      onRemove(pendingRemoval.path);
    } else {
      onRemoveParent?.(pendingRemoval.path);
    }

    setPendingRemoval(null);
  };

  return (
    <>
      <ModalBackdrop
        onClose={onClose}
        aria-label={t(I18nKey.HOME$MANAGE_WORKSPACES)}
      >
        <div
          data-testid="manage-workspaces-modal"
          className={cn(
            "flex flex-col bg-[var(--oh-surface)] border border-[var(--oh-border-input)] rounded-xl",
            modalWidthClassName("lg"),
            MODAL_MAX_WIDTH_VIEWPORT,
            "max-h-[70vh]",
          )}
        >
          <div className="flex items-center justify-between px-5 py-3 border-b border-[var(--oh-border-input)]">
            <BaseModalTitle
              className={modalTitleSmClassName}
              title={t(I18nKey.HOME$MANAGE_WORKSPACES)}
            />
          </div>

          <div
            className="flex-1 overflow-auto custom-scrollbar-always"
            data-testid="manage-workspaces-list"
          >
            {!hasContent && (
              <p className="px-5 py-6 text-sm text-[var(--oh-text-secondary)] text-center">
                {t(I18nKey.HOME$MANAGE_WORKSPACES_EMPTY)}
              </p>
            )}

            {staticWorkspaces.length > 0 && (
              <ul>
                {staticWorkspaces.map((workspace) => (
                  <li
                    key={workspace.id}
                    className="flex items-center gap-3 px-5 py-2 border-b border-[var(--oh-border-subtle)] last:border-b-0"
                    data-testid={`manage-workspaces-row-${workspace.name}`}
                  >
                    <FolderIcon width={16} height={16} className="shrink-0" />
                    <div className="flex flex-col min-w-0 flex-1">
                      <span className="text-sm text-white truncate">
                        {workspace.name}
                      </span>
                      <span className="text-xs text-[var(--oh-muted)] truncate">
                        {workspace.path}
                      </span>
                    </div>
                    <button
                      type="button"
                      onClick={() =>
                        setPendingRemoval({
                          type: "workspace",
                          path: workspace.path,
                          text: t(I18nKey.HOME$REMOVE_WORKSPACE_CONFIRMATION, {
                            name: workspace.name,
                          }),
                        })
                      }
                      aria-label={t(I18nKey.HOME$REMOVE_WORKSPACE)}
                      data-testid={`manage-workspaces-remove-${workspace.name}`}
                      className="flex items-center gap-1 px-2 py-1 rounded text-xs text-[var(--oh-text-tertiary)] hover:bg-[var(--oh-interactive-hover)] hover:text-white cursor-pointer"
                    >
                      <CloseIcon width={12} height={12} />
                      <span>{t(I18nKey.HOME$REMOVE_WORKSPACE)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}

            {workspaceParents.length > 0 && (
              <div data-testid="manage-workspaces-parents-section">
                <div className="px-5 pt-3 pb-1 text-[11px] uppercase tracking-wide text-[var(--oh-muted)] font-semibold">
                  {t(I18nKey.HOME$WORKSPACE_PARENTS)}
                </div>
                <ul>
                  {workspaceParents.map((parent) => {
                    const children =
                      dynamicWorkspacesByParent.get(parent.path) ?? [];
                    return (
                      <li
                        key={parent.id}
                        className="border-b border-[var(--oh-border-subtle)] last:border-b-0"
                        data-testid={`manage-workspaces-parent-row-${parent.name}`}
                      >
                        <div className="flex items-center gap-3 px-5 py-2">
                          <FolderIcon
                            width={16}
                            height={16}
                            className="shrink-0"
                          />
                          <div className="flex flex-col min-w-0 flex-1">
                            <span className="text-sm text-white truncate">
                              {parent.name}
                            </span>
                            <span className="text-xs text-[var(--oh-muted)] truncate">
                              {parent.path}
                            </span>
                          </div>
                          {onRemoveParent && (
                            <button
                              type="button"
                              onClick={() =>
                                setPendingRemoval({
                                  type: "parent",
                                  path: parent.path,
                                  text: t(
                                    I18nKey.HOME$REMOVE_WORKSPACE_PARENT_CONFIRMATION,
                                    {
                                      name: parent.name,
                                      count: children.length,
                                    },
                                  ),
                                })
                              }
                              aria-label={t(
                                I18nKey.HOME$REMOVE_WORKSPACE_PARENT,
                              )}
                              data-testid={`manage-workspaces-remove-parent-${parent.name}`}
                              className="flex items-center gap-1 px-2 py-1 rounded text-xs text-[var(--oh-text-tertiary)] hover:bg-[var(--oh-interactive-hover)] hover:text-white cursor-pointer"
                            >
                              <CloseIcon width={12} height={12} />
                              <span>
                                {t(I18nKey.HOME$REMOVE_WORKSPACE_PARENT)}
                              </span>
                            </button>
                          )}
                        </div>
                        {children.length > 0 && (
                          <ul className="pb-2">
                            {children.map((child) => (
                              <li
                                key={child.id}
                                className="flex items-center gap-3 px-5 pl-10 py-1 text-xs text-[var(--oh-text-secondary)]"
                                data-testid={`manage-workspaces-child-${child.name}`}
                              >
                                <FolderIcon
                                  width={12}
                                  height={12}
                                  className="shrink-0 opacity-70"
                                />
                                <span className="truncate">{child.name}</span>
                              </li>
                            ))}
                          </ul>
                        )}
                      </li>
                    );
                  })}
                </ul>
              </div>
            )}
          </div>

          <div className="flex justify-end gap-2 px-5 py-3 border-t border-[var(--oh-border-input)]">
            <BrandButton
              type="button"
              variant="primary"
              onClick={onClose}
              testId="manage-workspaces-done"
            >
              {t(I18nKey.HOME$DONE)}
            </BrandButton>
          </div>
        </div>
      </ModalBackdrop>

      {pendingRemoval && (
        <ConfirmationModal
          text={pendingRemoval.text}
          onConfirm={handleConfirmRemoval}
          onCancel={() => setPendingRemoval(null)}
        />
      )}
    </>
  );
}
