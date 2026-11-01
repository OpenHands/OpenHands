import { I18nKey } from "#/i18n/declaration";
import type { ConversationTab } from "#/stores/conversation-store";

/**
 * DOM id of the drawer's tab panel, referenced by every tab's `aria-controls`.
 *
 * The desktop drawer and the mobile `/panel` route each render one tab
 * strip plus one panel, and never both at once, so a constant panel id is
 * unambiguous.
 */
export const CONVERSATION_TAB_PANEL_ID = "conversation-tab-panel";

/** Each tab's name: shown in the strip and naming the panel it controls. */
export const CONVERSATION_TAB_LABEL_KEYS: Record<ConversationTab, I18nKey> = {
  files: I18nKey.COMMON$FILES,
  commits: I18nKey.DIFF_VIEWER$COMMITS,
  planner: I18nKey.COMMON$PLANNER,
  tasklist: I18nKey.COMMON$TASK_LIST,
  terminal: I18nKey.COMMON$TERMINAL,
  browser: I18nKey.COMMON$BROWSER,
  usage: I18nKey.COMMON$USAGE,
};
