/**
 * DOM ids wiring the drawer's tab strip to the panel it controls.
 *
 * The desktop drawer and the mobile `/panel` route each render one tab
 * strip plus one panel, and never both at once, so a constant panel id is
 * unambiguous.
 */
export const CONVERSATION_TAB_PANEL_ID = "conversation-tab-panel";

/** Id of the tab button for `tabValue` — the panel's `aria-labelledby`. */
export const conversationTabId = (tabValue: string) =>
  `conversation-tab-${tabValue}`;
