/**
 * Boolean conversation state carried in server-side conversation tags.
 *
 * Pinned and archived used to live only in browser localStorage
 * (`pinned-conversations` / `archived-conversations` stores), so they never
 * left the device that set them. They are now stamped as tags on the
 * conversation itself (`pinned=true` / `archived=true`), which the
 * agent-server stores on `ConversationInfo.tags` and the Cloud app-server
 * stores on the app-conversation. The sidebar derives both states from the
 * tags on the conversations it already fetches; the localStorage stores
 * remain only as a one-time migration source for pre-existing entries.
 *
 * Keys obey the backend tag-key rule (lowercase alphanumeric, see
 * `ConversationInfo.tags` in the SDK), hence the squashed forms.
 */
export const PINNED_CONVERSATION_TAG_KEY = "pinned";
export const ARCHIVED_CONVERSATION_TAG_KEY = "archived";

/** Value stamped for an enabled flag. Any other value counts as disabled. */
export const CONVERSATION_TAG_FLAG_VALUE = "true";

export function hasConversationTagFlag(
  tags: Record<string, string> | null | undefined,
  key: string,
): boolean {
  return tags?.[key] === CONVERSATION_TAG_FLAG_VALUE;
}

/**
 * Full tag map with the given flags applied, for backends whose PATCH
 * replaces the complete map (local agent-server). Returns a new object;
 * disabling a flag deletes the key rather than stamping a false value.
 */
export function withConversationTagFlags(
  tags: Record<string, string> | null | undefined,
  changes: Record<string, boolean>,
): Record<string, string> {
  const next: Record<string, string> = { ...(tags ?? {}) };
  for (const [key, enabled] of Object.entries(changes)) {
    if (enabled) {
      next[key] = CONVERSATION_TAG_FLAG_VALUE;
    } else {
      delete next[key];
    }
  }
  return next;
}

/**
 * Merge-patch fragment for backends that patch tags key-by-key (Cloud
 * app-server `PATCH /api/v1/app-conversations/{id}`): an enabled flag sets
 * the key, a disabled flag sends `null` to delete it. Keys not mentioned in
 * `changes` are left untouched server-side.
 */
export function conversationTagFlagPatch(
  changes: Record<string, boolean>,
): Record<string, string | null> {
  const patch: Record<string, string | null> = {};
  for (const [key, enabled] of Object.entries(changes)) {
    patch[key] = enabled ? CONVERSATION_TAG_FLAG_VALUE : null;
  }
  return patch;
}
