import React from "react";
import { useTranslation } from "react-i18next";
import { FolderBrowserModal } from "#/components/features/home/workspace-dropdown/folder-browser-modal";
import { ManageWorkspacesModal } from "#/components/features/home/workspace-dropdown/manage-workspaces-modal";
import { WorkspaceDropdown } from "#/components/features/home/workspace-dropdown/workspace-dropdown";
import {
  useAddWorkspaceParents,
  useAddWorkspaces,
  useRemoveWorkspace,
  useRemoveWorkspaceParent,
} from "#/hooks/mutation/use-local-workspaces-mutations";
import { I18nKey } from "#/i18n/declaration";
import type { LocalWorkspace, LocalWorkspaceParent } from "#/types/workspace";
import { getWorkspacesUnsupportedMessage } from "#/utils/workspaces-compatibility";
import { cn } from "#/utils/utils";

export interface KanbanWorkspacePickerProps {
  workspaces: LocalWorkspace[];
  parents: LocalWorkspaceParent[];
  workspaceParents: LocalWorkspaceParent[];
  selected: LocalWorkspace | null;
  isAllWorkspaces?: boolean;
  isLoading: boolean;
  listError: unknown;
  onChange: (workspace: LocalWorkspace | null) => void;
  onSelectAll?: () => void;
  className?: string;
}

export function KanbanWorkspacePicker({
  workspaces,
  parents,
  workspaceParents,
  selected,
  isAllWorkspaces = false,
  isLoading,
  listError,
  onChange,
  onSelectAll,
  className,
}: KanbanWorkspacePickerProps) {
  const { t } = useTranslation("openhands");
  const [isBrowserOpen, setIsBrowserOpen] = React.useState(false);
  const [isManageOpen, setIsManageOpen] = React.useState(false);
  const { mutate: addWorkspaces } = useAddWorkspaces();
  const { mutate: addWorkspaceParents } = useAddWorkspaceParents();
  const { mutate: removeWorkspace } = useRemoveWorkspace();
  const { mutate: removeWorkspaceParent } = useRemoveWorkspaceParent();
  const unsupported = getWorkspacesUnsupportedMessage(listError, t);
  const disabled =
    Boolean(unsupported) || (isLoading && workspaces.length === 0);

  return (
    <>
      <div
        className={cn(
          "flex w-full max-w-[22rem] items-center gap-2",
          className,
        )}
        data-testid="kanban-workspace-picker"
      >
        {onSelectAll ? (
          <button
            type="button"
            data-testid="kanban-all-workspaces"
            aria-pressed={isAllWorkspaces}
            onClick={onSelectAll}
            className={cn(
              "h-9 shrink-0 rounded-lg border px-2.5 text-xs font-medium",
              isAllWorkspaces
                ? "border-[var(--oh-border)] bg-[var(--oh-interactive-hover)] text-[var(--oh-foreground)]"
                : "border-[var(--oh-border)] text-[var(--oh-muted)] hover:bg-[var(--oh-interactive-hover)] hover:text-[var(--oh-foreground)]",
            )}
          >
            {t(I18nKey.KANBAN$ALL_WORKSPACES)}
          </button>
        ) : null}
        <div className="min-w-0 flex-1">
          <WorkspaceDropdown
            key={
              isAllWorkspaces
                ? "all-workspaces"
                : (selected?.path ?? "empty-workspace-selection")
            }
            workspaces={workspaces}
            parents={parents}
            value={isAllWorkspaces ? null : selected}
            placeholder={
              unsupported
                ? t(I18nKey.HOME$WORKSPACES_UNSUPPORTED_PLACEHOLDER)
                : isAllWorkspaces
                  ? t(I18nKey.KANBAN$ALL_WORKSPACES)
                  : disabled
                    ? t(I18nKey.HOME$LOADING)
                    : t(I18nKey.HOME$WORKSPACE_PLACEHOLDER)
            }
            disabled={disabled}
            disabledTooltip={unsupported}
            showManage={workspaces.length > 0 || workspaceParents.length > 0}
            className="w-full"
            onChange={onChange}
            onAddClick={() => setIsBrowserOpen(true)}
            onManageClick={() => setIsManageOpen(true)}
          />
        </div>
      </div>
      <FolderBrowserModal
        isOpen={isBrowserOpen}
        onClose={() => setIsBrowserOpen(false)}
        onAdd={(items) => {
          const lastAdded = items[items.length - 1];
          addWorkspaces(items, {
            onSuccess: () => {
              if (lastAdded) onChange(lastAdded);
            },
          });
        }}
        onAddParent={(items) => addWorkspaceParents(items)}
      />
      <ManageWorkspacesModal
        isOpen={isManageOpen}
        workspaces={workspaces}
        workspaceParents={workspaceParents}
        onClose={() => setIsManageOpen(false)}
        onRemove={(path) => {
          if (selected?.path === path) onChange(null);
          removeWorkspace(path);
        }}
        onRemoveParent={(path) => {
          if (selected?.parentPath === path) onChange(null);
          removeWorkspaceParent(path);
        }}
      />
    </>
  );
}
