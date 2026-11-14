import { I18nKey } from "#/i18n/declaration";
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

/**
 * Projects mark a finished conversation with `resolved=<who>` (`user` when the user
 * resolved it, `auto` when the Agent Server resolved it from a passing
 * critic score). Reopening removes the key; the Agent Server also removes it
 * when a new user message arrives.
 */
export const RESOLVED_CONVERSATION_TAG_KEY = "resolved";

export type ConversationResolution = "user" | "auto";

export function getConversationResolution(
  tags: Record<string, string> | null | undefined,
): ConversationResolution | null {
  const value = tags?.[RESOLVED_CONVERSATION_TAG_KEY];
  return value === "user" || value === "auto" ? value : null;
}

/** Why the server resolved a conversation, from its `resolvenote` tag. */
export const RESOLVE_NOTE_TAG_KEY = "resolvenote";

export type AutoResolveNote =
  /** The conversation's own agent resolved it (`resolve_conversation`). */
  | { source: "agent" }
  /** Rounded percent: the judge's "nothing left" probability or a critic score. */
  | { source: "judge" | "critic"; percent: number };

/**
 * Reads a `resolvenote`: `agent` (the conversation's agent resolved it),
 * `0.93 nothing` (the resolution judge) or `0.95 critic` (a critic score).
 * Returns null for anything else.
 */
export function parseAutoResolveNote(
  note: string | null | undefined,
): AutoResolveNote | null {
  const text = note?.trim();
  if (text === "agent") return { source: "agent" };
  const match = text?.match(/^([01](?:\.\d+)?) (\w+)$/);
  if (!match) return null;
  const probability = Number(match[1]);
  if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
    return null;
  }
  return {
    percent: Math.round(probability * 100),
    source: match[2] === "critic" ? "critic" : "judge",
  };
}

/** The i18n key and values that say why a conversation resolved itself. */
export function autoResolveMessage(
  note: AutoResolveNote,
): [I18nKey, Record<string, number>] {
  if (note.source === "agent")
    return [I18nKey.PROJECTS$AUTO_RESOLVED_AGENT, {}];
  return [
    note.source === "critic"
      ? I18nKey.PROJECTS$AUTO_RESOLVED_CRITIC
      : I18nKey.PROJECTS$AUTO_RESOLVED,
    { percent: note.percent },
  ];
}
