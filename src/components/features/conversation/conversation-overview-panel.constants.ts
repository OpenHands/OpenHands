export const CONVERSATION_OVERVIEW_AUTOMATIONS_PATH = "/automations";
export const CONVERSATION_OVERVIEW_SKILLS_PATH = "/skills";
export const CONVERSATION_OVERVIEW_MCP_PATH = "/mcp";

export const CONVERSATION_OVERVIEW_PANEL_WIDTH_PX = 240;
export const CONVERSATION_OVERVIEW_COLUMN_MIN_GAP_PX = 8;
/** Horizontal padding on the overview column (`pl-3` + `pr-4`). */
export const CONVERSATION_OVERVIEW_COLUMN_LEFT_PADDING_PX = 12;
export const CONVERSATION_OVERVIEW_COLUMN_RIGHT_PADDING_PX = 16;
export const CONVERSATION_OVERVIEW_COLUMN_HORIZONTAL_PADDING_PX =
  CONVERSATION_OVERVIEW_COLUMN_LEFT_PADDING_PX +
  CONVERSATION_OVERVIEW_COLUMN_RIGHT_PADDING_PX;
/** Minimum chat-thread width before the overview panel is hidden entirely. */
export const CONVERSATION_OVERVIEW_MIN_THREAD_WIDTH_PX = 320;

export const CONVERSATION_OVERVIEW_PANEL_TRANSITION = {
  duration: 0.2,
  ease: "easeInOut" as const,
};

export const CONVERSATION_OVERVIEW_COLUMN_WIDTH_PX =
  CONVERSATION_OVERVIEW_PANEL_WIDTH_PX +
  CONVERSATION_OVERVIEW_COLUMN_HORIZONTAL_PADDING_PX;

export function hasEnoughOverviewLayoutSpace(containerWidth: number): boolean {
  return (
    containerWidth >=
    CONVERSATION_OVERVIEW_COLUMN_WIDTH_PX +
      CONVERSATION_OVERVIEW_COLUMN_MIN_GAP_PX +
      CONVERSATION_OVERVIEW_MIN_THREAD_WIDTH_PX
  );
}

export const CONVERSATION_OVERVIEW_ADD_QUERY_PARAM = "add";
export const CONVERSATION_OVERVIEW_ADD_QUERY_VALUE = "1";

export const CONVERSATION_OVERVIEW_AUTOMATIONS_ADD_PATH =
  "/automations/dashboard?add=1";
export const CONVERSATION_OVERVIEW_SKILLS_ADD_PATH = "/skills?add=1";
export const CONVERSATION_OVERVIEW_MCP_ADD_PATH = "/mcp?add=1";

export function buildConversationOverviewAddPath(path: string): string {
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}${CONVERSATION_OVERVIEW_ADD_QUERY_PARAM}=${CONVERSATION_OVERVIEW_ADD_QUERY_VALUE}`;
}

export function hasConversationOverviewAddIntent(search: string): boolean {
  const normalized = search.startsWith("?") ? search.slice(1) : search;
  const params = new URLSearchParams(normalized);
  return (
    params.get(CONVERSATION_OVERVIEW_ADD_QUERY_PARAM) ===
    CONVERSATION_OVERVIEW_ADD_QUERY_VALUE
  );
}
