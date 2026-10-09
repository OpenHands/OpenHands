import { useConversationLocalStorageState } from "#/utils/conversation-local-storage";
import {
  useConversationStore,
  type ConversationTab,
} from "#/stores/conversation-store";
import { useOptionalConversationId } from "#/hooks/use-conversation-id";

/**
 * Custom hook for selecting conversation tabs with consistent behavior.
 *
 * Handles panel visibility and tab toggling logic. The selected tab is
 * persisted per conversation (so users land on the same tab when they
 * come back), but the drawer's open/closed state is intentionally
 * session-only — see `useConversationStore` for the rationale.
 */
export function useSelectConversationTab() {
  const { conversationId } = useOptionalConversationId();
  const {
    selectedTab,
    isRightPanelShown,
    setHasRightPanelToggled,
    setIsOverviewPanelShown,
    setIsRightPanelShown,
    setSelectedTab,
    setCommitsAutoExpandSection,
  } = useConversationStore();

  const { setSelectedTab: setPersistedSelectedTab, setRightPanelShown } =
    useConversationLocalStorageState(conversationId ?? "");

  const setPanelOpen = (open: boolean) => {
    setHasRightPanelToggled(open);
    setIsRightPanelShown(open);
    setRightPanelShown?.(open);
  };

  const onTabChange = (value: ConversationTab | null) => {
    setSelectedTab(value);
    setPersistedSelectedTab(value);
  };

  /**
   * Selects a tab with proper panel visibility handling.
   * - If clicking the same active tab while panel is open, closes the panel
   * - If clicking a different tab or panel is closed, opens panel and selects tab
   */
  const selectTab = (tab: ConversationTab) => {
    if (selectedTab === tab && isRightPanelShown) {
      setPanelOpen(false);
    } else {
      onTabChange(tab);
      if (!isRightPanelShown) {
        setPanelOpen(true);
      }
      setIsOverviewPanelShown(false);
    }
  };

  /**
   * Navigates to a tab without toggle behavior.
   * Always shows the panel and selects the tab, even if already selected.
   * Use this for "View" or "Read More" buttons that should always navigate.
   */
  const navigateToTab = (tab: ConversationTab) => {
    onTabChange(tab);
    setPanelOpen(true);
    setIsOverviewPanelShown(false);
  };

  const navigateToChanges = () => {
    setCommitsAutoExpandSection("uncommitted");
    navigateToTab("commits");
  };

  const navigateToCommits = () => {
    setCommitsAutoExpandSection(null);
    navigateToTab("commits");
  };

  /**
   * Checks if a specific tab is currently active (selected and panel is visible).
   */
  const isTabActive = (tab: ConversationTab) =>
    isRightPanelShown && selectedTab === tab;

  return {
    selectTab,
    navigateToTab,
    navigateToChanges,
    navigateToCommits,
    isTabActive,
    onTabChange,
    selectedTab,
    isRightPanelShown,
  };
}
