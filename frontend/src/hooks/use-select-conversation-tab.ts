import { useConversationLocalStorageState } from "#/utils/conversation-local-storage";
import {
  useConversationStore,
  type ConversationTab,
} from "#/stores/conversation-store";
import { useConversationId } from "#/hooks/use-conversation-id";

/**
 * Custom hook for selecting conversation tabs with consistent behavior.
 * Handles panel visibility, state persistence, and tab toggling logic.
 */
export function useSelectConversationTab() {
  const { conversationId } = useConversationId();
  const {
    selectedTab,
    isRightPanelShown,
    setHasRightPanelToggled,
    setSelectedTab,
  } = useConversationStore();

  const {
    setSelectedTab: setPersistedSelectedTab,
    setRightPanelShown: setPersistedRightPanelShown,
  } = useConversationLocalStorageState(conversationId);

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
      // If clicking the same active tab, close the drawer
      setHasRightPanelToggled(false);
      setPersistedRightPanelShown(false);
    } else {
      // If clicking a different tab or drawer is closed, open drawer and select tab
      onTabChange(tab);
      if (!isRightPanelShown) {
        setHasRightPanelToggled(true);
        setPersistedRightPanelShown(true);
      }
    }
  };

  /**
   * Navigates to a tab without toggle behavior.
   * Always shows the panel and selects the tab, even if already selected.
   * Use this for "View" or "Read More" buttons that should always navigate.
   */
  const navigateToTab = (tab: ConversationTab) => {
    onTabChange(tab);
    if (!isRightPanelShown) {
      setHasRightPanelToggled(true);
      setPersistedRightPanelShown(true);
    }
  };

  /**
   * Checks if a specific tab is currently active (selected and panel is visible).
   */
  const isTabActive = (tab: ConversationTab) =>
    isRightPanelShown && selectedTab === tab;

  return {
    selectTab,
    navigateToTab,
    isTabActive,
    onTabChange,
    selectedTab,
    isRightPanelShown,
  };
}
