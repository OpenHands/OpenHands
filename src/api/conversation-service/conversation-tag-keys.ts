/**
 * Wire-level conversation tag keys.
 *
 * Extracted so both the conversation serializer and the tag-display helpers
 * can read them without importing each other, or the adapter that used to own
 * them. Keys are constrained to `^[a-z0-9]+$` by the agent-server validator.
 */

export const ACP_SERVER_TAG_KEY = "acpserver";

export const AUTOMATION_TRIGGER_TAG_KEY = "automationtrigger";
export const AUTOMATION_ID_TAG_KEY = "automationid";
export const AUTOMATION_NAME_TAG_KEY = "automationname";
export const AUTOMATION_RUN_ID_TAG_KEY = "automationrunid";

/**
 * Tag keys stamped on conversations created by automation runs (see the SDK's
 * `RemoteWorkspace.default_conversation_tags`). The presence of any of these
 * marks a conversation as automation-born.
 */
export const AUTOMATION_TAG_KEYS: readonly string[] = [
  AUTOMATION_TRIGGER_TAG_KEY,
  AUTOMATION_ID_TAG_KEY,
  AUTOMATION_NAME_TAG_KEY,
  AUTOMATION_RUN_ID_TAG_KEY,
];
