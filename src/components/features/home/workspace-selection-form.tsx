import React from "react";
import { useTranslation } from "react-i18next";

import { useCreateConversation } from "#/hooks/mutation/use-create-conversation";
import { useNavigation } from "#/context/navigation-context";
import { useIsCreatingConversation } from "#/hooks/use-is-creating-conversation";
import {
  useAddWorkspaces,
  useAddWorkspaceParents,
  useRemoveWorkspace,
  useRemoveWorkspaceParent,
} from "#/hooks/mutation/use-local-workspaces-mutations";
import { useLocalWorkspaces } from "#/hooks/query/use-local-workspaces";
import { useResolvedWorkspaces } from "#/hooks/query/use-resolved-workspaces";
import { LocalWorkspace } from "#/types/workspace";
import { I18nKey } from "#/i18n/declaration";
import FolderIcon from "#/icons/folder.svg?react";
import { getWorkspacesUnsupportedMessage } from "#/utils/workspaces-compatibility";

import { BrandButton } from "../settings/brand-button";
import { WorkspaceDropdown } from "./workspace-dropdown/workspace-dropdown";
import { FolderBrowserModal } from "./workspace-dropdown/folder-browser-modal";
import { ManageWorkspacesModal } from "./workspace-dropdown/manage-workspaces-modal";

interface WorkspaceSelectionFormProps {
  isLoadingSettings?: boolean;
  /**
   * When provided, the form skips its own conversation creation + navigation
   * and just calls back with the selected workspace. Callers (e.g. the home
   * launcher dialog) use this to capture the selection and create the
   * conversation themselves once the user submits a message. The button label
   * also flips from "Launch" to "Confirm" so the action matches the new flow.
   */
  onConfirm?: (workspace: LocalWorkspace) => void;
}

export const HOME_SELECTED_WORKSPACE_PATH_KEY =
  "oh:home-selected-workspace-path";

function getStoredSelectedWorkspacePath(): string | null {
  if (typeof window === "undefined") return null;

  try {
    const path = window.sessionStorage.getItem(
      HOME_SELECTED_WORKSPACE_PATH_KEY,
    );
    return path && path.length > 0 ? path : null;
  } catch {
    return null;
  }
}

function setStoredSelectedWorkspacePath(path: string | null): void {
  if (typeof window === "undefined") return;

  try {
    if (path) {
      window.sessionStorage.setItem(HOME_SELECTED_WORKSPACE_PATH_KEY, path);
    } else {
      window.sessionStorage.removeItem(HOME_SELECTED_WORKSPACE_PATH_KEY);
    }
  } catch {
    // sessionStorage may be unavailable in private browsing contexts.
  }
}

export function WorkspaceSelectionForm({
  isLoadingSettings = false,
  onConfirm,
}: WorkspaceSelectionFormProps) {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();

  const { data: workspacesData, error: workspacesError } = useLocalWorkspaces();
  const workspaceParents = workspacesData?.workspaceParents ?? [];
  const { mutate: addWorkspaces } = useAddWorkspaces();
  const { mutate: removeWorkspace } = useRemoveWorkspace();
  const { mutate: addWorkspaceParents } = useAddWorkspaceParents();
  const { mutate: removeWorkspaceParent } = useRemoveWorkspaceParent();
  const {
    workspaces,
    parents: resolvedParents,
    isLoading: isLoadingWorkspaces,
    isError: hasWorkspaceError,
    error: resolvedWorkspacesError,
  } = useResolvedWorkspaces();
  const workspacesUnsupportedMessage = getWorkspacesUnsupportedMessage(
    workspacesError ?? resolvedWorkspacesError,
    t,
  );
  const [selectedWorkspace, setSelectedWorkspace] =
    React.useState<LocalWorkspace | null>(null);
  const [isBrowserOpen, setIsBrowserOpen] = React.useState(false);
  const [isManageOpen, setIsManageOpen] = React.useState(false);

  const {
    mutate: createConversation,
    isPending,
    isSuccess,
  } = useCreateConversation();
  const isCreatingConversationElsewhere = useIsCreatingConversation();
  const isCreatingConversation =
    isPending || isSuccess || isCreatingConversationElsewhere;

  const handleWorkspaceChange = React.useCallback(
    (workspace: LocalWorkspace | null) => {
      setSelectedWorkspace(workspace);
      setStoredSelectedWorkspacePath(workspace?.path ?? null);
    },
    [],
  );

  React.useEffect(() => {
    const storedPath = getStoredSelectedWorkspacePath();
    if (!storedPath) return;

    const restoredWorkspace = workspaces.find((w) => w.path === storedPath);
    if (restoredWorkspace) {
      setSelectedWorkspace((current) =>
        current?.path === restoredWorkspace.path ? current : restoredWorkspace,
      );
      return;
    }

    if (
      !isLoadingWorkspaces &&
      !hasWorkspaceError &&
      !workspacesUnsupportedMessage
    ) {
      setStoredSelectedWorkspacePath(null);
      setSelectedWorkspace((current) =>
        current?.path === storedPath ? null : current,
      );
    }
  }, [
    hasWorkspaceError,
    isLoadingWorkspaces,
    workspaces,
    workspacesUnsupportedMessage,
  ]);

  const showWorkspaceStatus =
    workspaceParents.length > 0 || Boolean(workspacesUnsupportedMessage);
  let workspaceStatusText: string | null = null;
  if (workspacesUnsupportedMessage) {
    workspaceStatusText = workspacesUnsupportedMessage;
  } else if (isLoadingWorkspaces) {
    workspaceStatusText = t(I18nKey.HOME$LOADING);
  } else if (hasWorkspaceError) {
    workspaceStatusText = t(I18nKey.HOME$WORKSPACE_SCAN_ERROR);
  }
  const isDropdownDisabled =
    Boolean(workspacesUnsupportedMessage) ||
    isLoadingSettings ||
    (isLoadingWorkspaces && workspaces.length === 0);

  const handleLaunch = () => {
    if (!selectedWorkspace) return;
    if (onConfirm) {
      onConfirm(selectedWorkspace);
      return;
    }
    createConversation(
      { workingDir: selectedWorkspace.path, entryPoint: "home_workspace_form" },
      {
        onSuccess: (data) => navigate(`/conversations/${data.conversation_id}`),
      },
    );
  };

  return (
    <div className="flex flex-col">
      {/* Skip the in-form "Workspaces" header in dialog mode — the dialog
          already shows an "Open Workspace" title, so this would be redundant. */}
      {!onConfirm && (
        <div className="flex items-center gap-[10px] pb-4">
          <FolderIcon width={24} height={24} />
          <span className="leading-5 font-bold text-base text-white">
            {t(I18nKey.HOME$WORKSPACES_TAB)}
          </span>
        </div>
      )}

      <div className="flex flex-col gap-[10px] pb-4">
        <WorkspaceDropdown
          workspaces={workspaces}
          parents={resolvedParents}
          value={selectedWorkspace}
          placeholder={
            workspacesUnsupportedMessage
              ? t(I18nKey.HOME$WORKSPACES_UNSUPPORTED_PLACEHOLDER)
              : isDropdownDisabled
                ? t(I18nKey.HOME$LOADING)
                : undefined
          }
          disabled={isDropdownDisabled}
          disabledTooltip={workspacesUnsupportedMessage}
          showManage={workspaces.length > 0 || workspaceParents.length > 0}
          onChange={handleWorkspaceChange}
          onAddClick={() => setIsBrowserOpen(true)}
          onManageClick={() => setIsManageOpen(true)}
          className="max-w-auto"
          key={selectedWorkspace?.path ?? "empty-workspace-selection"}
        />

        {showWorkspaceStatus && workspaceStatusText && (
          <p
            className="px-1 text-xs text-[var(--oh-text-secondary)]"
            data-testid="workspace-status-message"
          >
            {workspaceStatusText}
          </p>
        )}
      </div>

      <BrandButton
        testId="workspace-launch-button"
        variant="primary"
        type="button"
        isDisabled={
          !selectedWorkspace ||
          (!onConfirm && isCreatingConversation) ||
          isLoadingSettings
        }
        onClick={handleLaunch}
        className={
          onConfirm ? "w-full" : "w-auto absolute bottom-5 left-5 right-5"
        }
      >
        {onConfirm
          ? t(I18nKey.BUTTON$CONFIRM)
          : !isCreatingConversation
            ? t(I18nKey.BUTTON$LAUNCH)
            : t(I18nKey.HOME$LOADING)}
      </BrandButton>

      <FolderBrowserModal
        isOpen={isBrowserOpen}
        onClose={() => setIsBrowserOpen(false)}
        onAdd={(items) => {
          // Auto-select the newly added workspace so the add action gives
          // immediate feedback. Selection happens in the per-call onSuccess:
          // the hook-level onSuccess awaits the list refetch, so the restore
          // effect above sees the new entry and won't clear the selection.
          const lastAdded = items[items.length - 1];
          addWorkspaces(items, {
            onSuccess: () => {
              if (lastAdded) handleWorkspaceChange(lastAdded);
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
          if (selectedWorkspace?.path === path) {
            handleWorkspaceChange(null);
          }
          removeWorkspace(path);
        }}
        onRemoveParent={(path) => {
          if (selectedWorkspace?.parentPath === path) {
            handleWorkspaceChange(null);
          }
          removeWorkspaceParent(path);
        }}
      />
    </div>
  );
}
