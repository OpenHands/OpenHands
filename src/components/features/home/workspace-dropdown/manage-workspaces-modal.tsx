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
import { StyledTooltip } from "#/components/shared/buttons/styled-tooltip";
import { I18nKey } from "#/i18n/declaration";
import { LocalWorkspace, LocalWorkspaceParent } from "#/types/workspace";
import { cn } from "#/utils/utils";
import { modalTitleSmClassName } from "#/utils/modal-classes";
import { formControlFieldClassName } from "#/utils/form-control-classes";
import { getWorkspaceRenameUnsupportedMessage } from "#/utils/workspaces-compatibility";
import FolderIcon from "#/icons/folder.svg?react";
import CloseIcon from "#/icons/close.svg?react";
import CheckmarkIcon from "#/icons/checkmark.svg?react";
import EditIcon from "#/icons/u-edit.svg?react";

/** Product limit for a workspace label. The registry schema allows 256. */
export const MAX_WORKSPACE_NAME_LENGTH = 64;

type RenameHandler = (path: string, name: string) => Promise<unknown> | void;

interface ManageWorkspacesModalProps {
  isOpen: boolean;
  workspaces: LocalWorkspace[];
  workspaceParents?: LocalWorkspaceParent[];
  onClose: () => void;
  onRemove: (path: string) => void;
  onRemoveParent?: (path: string) => void;
  /**
   * Rename handlers. They may return a promise (consumers pass the mutation's
   * `mutateAsync`): the row stays in edit mode until it settles, so a failed
   * rename keeps the typed text.
   */
  onRename?: RenameHandler;
  onRenameParent?: RenameHandler;
}

type PendingRemoval =
  | { type: "workspace"; path: string; text: string }
  | { type: "parent"; path: string; text: string };

type EditingTarget = { kind: "workspace" | "parent"; path: string };

/** The value to save, or `null` when the input is not a usable label. */
function normaliseName(value: string): string | null {
  const trimmed = value.trim();
  if (!trimmed || trimmed.length > MAX_WORKSPACE_NAME_LENGTH) return null;
  if (/[\r\n]/.test(trimmed)) return null;
  return trimmed;
}

interface NameEditorProps {
  idPrefix: string;
  value: string;
  isInvalid: boolean;
  canSave: boolean;
  isSaving: boolean;
  onChange: (value: string) => void;
  onSave: () => void;
  onCancel: () => void;
}

function NameEditor({
  idPrefix,
  value,
  isInvalid,
  canSave,
  isSaving,
  onChange,
  onSave,
  onCancel,
}: NameEditorProps) {
  const { t } = useTranslation("openhands");
  const inputRef = React.useRef<HTMLInputElement>(null);
  const ruleId = `${idPrefix}-rule`;

  // Edit mode is user-initiated, so focusing the field here is expected.
  React.useEffect(() => {
    inputRef.current?.focus();
    inputRef.current?.select();
  }, []);

  return (
    <form
      className="flex flex-col gap-1"
      onSubmit={(event) => {
        event.preventDefault();
        if (canSave) onSave();
      }}
    >
      <div className="flex items-center gap-1">
        <input
          ref={inputRef}
          type="text"
          value={value}
          maxLength={MAX_WORKSPACE_NAME_LENGTH}
          aria-label={t(I18nKey.HOME$WORKSPACE_NAME_LABEL)}
          aria-invalid={isInvalid}
          aria-describedby={isInvalid ? ruleId : undefined}
          data-testid={`${idPrefix}-input`}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Escape") {
              event.preventDefault();
              onCancel();
            }
          }}
          className={cn(formControlFieldClassName, "h-8 py-0 text-sm")}
        />
        <button
          type="button"
          disabled={!canSave}
          aria-label={t(I18nKey.BUTTON$SAVE)}
          data-testid={`${idPrefix}-save`}
          onClick={onSave}
          className="flex items-center justify-center rounded p-1 text-text-tertiary hover:bg-interactive-hover hover:text-contrast cursor-pointer disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent"
        >
          <CheckmarkIcon width={12} height={12} />
        </button>
        <button
          type="button"
          disabled={isSaving}
          aria-label={t(I18nKey.BUTTON$CANCEL)}
          data-testid={`${idPrefix}-cancel`}
          onClick={onCancel}
          className="flex items-center justify-center rounded p-1 text-text-tertiary hover:bg-interactive-hover hover:text-contrast cursor-pointer disabled:cursor-not-allowed disabled:opacity-40"
        >
          <CloseIcon
            width={12}
            height={12}
            className="text-contrast dark:text-white"
          />
        </button>
      </div>
      {isInvalid && (
        <p id={ruleId} className="text-xs text-danger">
          {t(I18nKey.HOME$WORKSPACE_NAME_RULE)}
        </p>
      )}
    </form>
  );
}

export function ManageWorkspacesModal({
  isOpen,
  workspaces,
  workspaceParents = [],
  onClose,
  onRemove,
  onRemoveParent,
  onRename,
  onRenameParent,
}: ManageWorkspacesModalProps) {
  const { t } = useTranslation("openhands");
  const [pendingRemoval, setPendingRemoval] =
    React.useState<PendingRemoval | null>(null);
  const [editing, setEditing] = React.useState<EditingTarget | null>(null);
  const [draftName, setDraftName] = React.useState("");
  const [isSaving, setIsSaving] = React.useState(false);

  const renameUnsupportedMessage = getWorkspaceRenameUnsupportedMessage(t);

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

  const isEditing = (kind: EditingTarget["kind"], path: string) =>
    editing?.kind === kind && editing.path === path;

  const startEditing = (target: EditingTarget, currentName: string) => {
    setEditing(target);
    setDraftName(currentName);
  };

  const cancelEditing = () => {
    setEditing(null);
    setIsSaving(false);
  };

  const currentNameOf = (target: EditingTarget) =>
    target.kind === "workspace"
      ? staticWorkspaces.find((w) => w.path === target.path)?.name
      : workspaceParents.find((p) => p.path === target.path)?.name;

  const commitEditing = async () => {
    if (!editing) return;
    const name = normaliseName(draftName);
    if (!name || name === currentNameOf(editing)) return;

    const handler = editing.kind === "workspace" ? onRename : onRenameParent;
    if (!handler) return;

    setIsSaving(true);
    try {
      await handler(editing.path, name);
      setEditing(null);
    } catch {
      // The mutation surfaces its own error toast; staying in edit mode keeps
      // the typed text so the user can retry or cancel.
    } finally {
      setIsSaving(false);
    }
  };

  const renderRenameControl = (
    target: EditingTarget,
    name: string,
    enabled: boolean,
  ) => {
    if (!enabled || isEditing(target.kind, target.path)) return null;

    const isWorkspaceRow = target.kind === "workspace";
    const label = isWorkspaceRow
      ? t(I18nKey.HOME$RENAME_WORKSPACE)
      : t(I18nKey.HOME$RENAME_WORKSPACE_PARENT);
    const testId = isWorkspaceRow
      ? `manage-workspaces-rename-${name}`
      : `manage-workspaces-rename-parent-${name}`;

    const button = (
      <button
        type="button"
        disabled={Boolean(renameUnsupportedMessage)}
        onClick={() => startEditing(target, name)}
        aria-label={`${label}: ${name}`}
        data-testid={testId}
        className="flex shrink-0 items-center justify-center rounded p-1 text-text-tertiary hover:bg-interactive-hover hover:text-contrast cursor-pointer opacity-100 transition-opacity md:opacity-0 md:group-hover:opacity-100 focus-visible:opacity-100 disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:bg-transparent"
      >
        <EditIcon width={12} height={12} />
      </button>
    );

    if (!renameUnsupportedMessage) return button;

    return (
      <StyledTooltip content={renameUnsupportedMessage} placement="top">
        <span className="inline-flex shrink-0">{button}</span>
      </StyledTooltip>
    );
  };

  const renderNameColumn = (
    target: EditingTarget,
    item: { name: string; path: string },
  ) => {
    if (!isEditing(target.kind, target.path)) {
      return (
        <span className="text-sm text-contrast truncate">{item.name}</span>
      );
    }

    const editorName = normaliseName(draftName);
    return (
      <NameEditor
        idPrefix={`manage-workspaces-rename-${target.kind}`}
        value={draftName}
        isInvalid={editorName === null}
        canSave={
          !isSaving &&
          editorName !== null &&
          editorName !== currentNameOf(target)
        }
        isSaving={isSaving}
        onChange={setDraftName}
        onSave={commitEditing}
        onCancel={cancelEditing}
      />
    );
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
            "flex flex-col bg-surface border border-border-input rounded-xl",
            modalWidthClassName("lg"),
            MODAL_MAX_WIDTH_VIEWPORT,
            "workspace-manager-dialog",
          )}
        >
          <div className="flex items-center justify-between px-5 py-3 border-b border-border-input">
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
              <p className="px-5 py-6 text-sm text-text-secondary text-center">
                {t(I18nKey.HOME$MANAGE_WORKSPACES_EMPTY)}
              </p>
            )}

            {staticWorkspaces.length > 0 && (
              <ul>
                {staticWorkspaces.map((workspace) => (
                  <li
                    key={workspace.id}
                    className="group flex items-center gap-3 px-5 py-2 border-b border-border-subtle last:border-b-0"
                    data-testid={`manage-workspaces-row-${workspace.name}`}
                  >
                    <FolderIcon width={16} height={16} className="shrink-0" />
                    <div className="flex flex-col min-w-0 flex-1">
                      {renderNameColumn(
                        { kind: "workspace", path: workspace.path },
                        workspace,
                      )}
                      <span className="text-xs text-muted truncate">
                        {workspace.path}
                      </span>
                    </div>
                    {renderRenameControl(
                      { kind: "workspace", path: workspace.path },
                      workspace.name,
                      Boolean(onRename),
                    )}
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
                      className="flex items-center gap-1 px-2 py-1 rounded text-xs text-text-tertiary hover:bg-interactive-hover hover:text-contrast cursor-pointer"
                    >
                      <CloseIcon
                        width={12}
                        height={12}
                        className="text-contrast dark:text-white"
                      />
                      <span>{t(I18nKey.HOME$REMOVE_WORKSPACE)}</span>
                    </button>
                  </li>
                ))}
              </ul>
            )}

            {workspaceParents.length > 0 && (
              <div data-testid="manage-workspaces-parents-section">
                <div className="px-5 pt-3 pb-1 workspace-section-label uppercase tracking-wide text-muted font-semibold">
                  {t(I18nKey.HOME$WORKSPACE_PARENTS)}
                </div>
                <ul>
                  {workspaceParents.map((parent) => {
                    const children =
                      dynamicWorkspacesByParent.get(parent.path) ?? [];
                    return (
                      <li
                        key={parent.id}
                        className="border-b border-border-subtle last:border-b-0"
                        data-testid={`manage-workspaces-parent-row-${parent.name}`}
                      >
                        <div className="group flex items-center gap-3 px-5 py-2">
                          <FolderIcon
                            width={16}
                            height={16}
                            className="shrink-0"
                          />
                          <div className="flex flex-col min-w-0 flex-1">
                            {renderNameColumn(
                              { kind: "parent", path: parent.path },
                              parent,
                            )}
                            <span className="text-xs text-muted truncate">
                              {parent.path}
                            </span>
                          </div>
                          {renderRenameControl(
                            { kind: "parent", path: parent.path },
                            parent.name,
                            Boolean(onRenameParent),
                          )}
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
                              className="flex items-center gap-1 px-2 py-1 rounded text-xs text-text-tertiary hover:bg-interactive-hover hover:text-contrast cursor-pointer"
                            >
                              <CloseIcon
                                width={12}
                                height={12}
                                className="text-contrast dark:text-white"
                              />
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
                                className="flex items-center gap-3 px-5 pl-10 py-1 text-xs text-text-secondary"
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

          <div className="flex justify-end gap-2 px-5 py-3 border-t border-border-input">
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
