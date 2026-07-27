import type { BackendKind } from "#/api/backend-registry/types";
import type { WorkspaceMode } from "#/api/conversation-metadata-store";
import { I18nKey } from "#/i18n/declaration";

export function getWorkspaceModeI18nKey(
  mode: WorkspaceMode,
  backendKind: BackendKind,
): I18nKey {
  if (mode === "new_worktree") {
    return I18nKey.COMMON$WORKSPACE_MODE_NEW_WORKTREE;
  }
  return backendKind === "cloud"
    ? I18nKey.COMMON$WORKSPACE_MODE_CLOUD_REPO
    : I18nKey.COMMON$WORKSPACE_MODE_LOCAL_REPO;
}
