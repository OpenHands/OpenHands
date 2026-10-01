/**
 * DOM id of the drawer's tab panel, referenced by every tab's `aria-controls`.
 *
 * The desktop drawer and the mobile `/panel` route each render one tab
 * strip plus one panel, and never both at once, so a constant panel id is
 * unambiguous.
 */
export const CONVERSATION_TAB_PANEL_ID = "conversation-tab-panel";
