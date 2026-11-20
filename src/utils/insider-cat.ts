export const INSIDER_CAT_PROJECTS_PATH = "/extensions/insider-cat/projects";
export const INSIDER_CAT_NEW_PATH = `${INSIDER_CAT_PROJECTS_PATH}/new`;

export const INSIDER_CAT_TAG = "smolpaws";
const INSIDER_CAT_VALUE = "insider";
export const INSIDER_ROLE_TAG = "insiderrole";
const INSIDER_CONTROLLER_ROLE = "controller";

/** Legacy controllers have only the Cat tag; children and explicit other roles do not qualify. */
export function isInsiderConversation(
  conversation:
    | {
        tags?: Record<string, string> | null;
        parent_conversation_id?: string | null;
      }
    | null
    | undefined,
): boolean {
  const role = conversation?.tags?.[INSIDER_ROLE_TAG];
  return (
    conversation?.tags?.[INSIDER_CAT_TAG] === INSIDER_CAT_VALUE &&
    !conversation.parent_conversation_id &&
    (role === undefined || role === INSIDER_CONTROLLER_ROLE)
  );
}

export function getInsiderConversationPath(conversationId: string): string {
  return `${INSIDER_CAT_PROJECTS_PATH}/conversations/${encodeURIComponent(conversationId)}`;
}
