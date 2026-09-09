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
  isLoading: boolean;
  listError: unknown;
  onChange: (workspace: LocalWorkspace | null) => void;
  className?: string;
}

export function KanbanWorkspacePicker({
  workspaces,
  parents,
  workspaceParents,
  selected,
  isLoading,
  listError,
  onChange,
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
        className={cn("w-full max-w-[14rem]", className)}
        data-testid="kanban-workspace-picker"
      >
        <WorkspaceDropdown
          key={selected?.path ?? "empty-workspace-selection"}
          workspaces={workspaces}
          parents={parents}
          value={selected}
          placeholder={
            unsupported
              ? t(I18nKey.HOME$WORKSPACES_UNSUPPORTED_PLACEHOLDER)
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
