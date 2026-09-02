import ConversationService from "#/api/conversation-service/conversation-service.api";
import {
  ConversationTab,
  useConversationStore,
} from "#/stores/conversation-store";
import { useFilesTabStore } from "#/stores/files-tab-store";
import type { CanvasUIAction } from "#/types/agent-server/core";
import { toFilesTabPath } from "#/utils/path-utils";

const VALID_TABS: ReadonlySet<ConversationTab> = new Set<ConversationTab>([
  "files",
  "browser",
  "terminal",
  "planner",
  "tasklist",
]);

// Mirrors src/hooks/use-select-conversation-tab.ts so a non-React caller (the
// WebSocket dispatch) gets the same "reveal the right panel if collapsed"
// behavior as in-app tab switches.
function navigateToTab(tab: ConversationTab) {
  const store = useConversationStore.getState();
  store.setSelectedTab(tab);
  if (!store.isRightPanelShown) {
    store.setHasRightPanelToggled(true);
    store.setIsRightPanelShown(true);
  }
}

function isValidTab(value: string): value is ConversationTab {
  return VALID_TABS.has(value as ConversationTab);
}

/**
 * Chat path click → same as agent `navigate_to_file`.
 * Optional `conversationId` tags the selection so FilesTab accepts it.
 */
export function openWorkspaceFile(
  path: string,
  conversationId?: string | null,
): void {
  const conversation = ConversationService.getCurrentConversation();
  handleCanvasUIAction(
    {
      kind: "CanvasUIAction",
      command: "navigate_to_file",
      path,
    } as CanvasUIAction,
    conversationId ?? conversation?.id ?? null,
  );
}

export function handleCanvasUIAction(
  action: CanvasUIAction,
  conversationId: string | null = null,
): void {
  switch (action.command) {
    case "navigate_to_file":
    case "show_preview": {
      navigateToTab("files");
      if (!action.path) return;

      const workingDir =
        ConversationService.getCurrentConversation()?.workspace?.working_dir;
      const path = toFilesTabPath(action.path, workingDir);
      if (!path) return;

      useFilesTabStore.getState().setSelectedPath(path, conversationId);
      return;
    }
    case "open_tab":
      if (action.tab === "vscode") {
        // The in-app VS Code tab was removed — on cloud backends VS Code
        // now opens in a new browser window via the link in the drawer tab
        // row. Route agent requests to Files so the drawer still opens.
        navigateToTab("files");
      } else if (action.tab && isValidTab(action.tab)) {
        navigateToTab(action.tab);
      } else if (action.tab) {
        // Surface unknown tab names so they're diagnosable from the browser
        // console rather than failing silently. Valid tabs are listed in
        // VALID_TABS above and mirror ConversationTab.
        console.warn(
          `[canvas_ui] Ignoring open_tab with unknown tab: ${action.tab}`,
        );
      }
      return;
  }
}
