import {
  AUTOMATION_NAME_TAG_KEY,
  getDefaultConversationTitle,
} from "#/api/agent-server-adapter";
import { getWorkItemDisplayTitleFromTags } from "#/utils/work-item-reference";

export function isPlaceholderConversationTitle(
  title: string | null | undefined,
  conversationId: string,
): boolean {
  return (
    !title?.trim() ||
    title.trim() === getDefaultConversationTitle(conversationId)
  );
}

/**
 * Separator the automation SDK uses between its configured name and the run
 * context in the machine title it writes (`"{name} — {context}"`, see the
 * SDK's `_build_conversation_title`). Recognized so that machine title can be
 * told apart from a user rename.
 */
export const AUTOMATION_TITLE_SEPARATOR = " \u2014 ";

/** Separator between the automation name and the work-item name. */
export const AUTOMATION_WORK_ITEM_SEPARATOR = ": ";

/**
 * Whether a server title is only the automation SDK's own machine title
 * (`"{automation name} — {context}"`, e.g. "🧬 CIPHER — 2026-11-18 19:21:42
 * UTC") rather than something a user set. Whitespace is normalized on both
 * sides because the SDK collapses the configured name before rendering it.
 * A null/blank title or a missing name is never a machine title.
 */
export function isAutomationMachineTitle(
  title: string | null | undefined,
  automationName: string | null | undefined,
): boolean {
  const value = title?.trim().replace(/\s+/g, " ");
  const name = automationName?.trim().replace(/\s+/g, " ");
  if (!value || !name) return false;
  return (
    value === name || value.startsWith(`${name}${AUTOMATION_TITLE_SEPARATOR}`)
  );
}

/**
 * The display name for an automation-born conversation that is linked to a
 * backlog work item: the automation's configured name prefixes the work-item
 * name, e.g. "🧬 CIPHER: GH-142 — Mission Provider Filtering". Returns null
 * unless *both* halves are present, so an ordinary conversation, a work-item
 * launch with no automation, and an automation run with no work item all keep
 * their existing name.
 */
export function composeAutomationWorkItemTitle(
  automationName: string | null | undefined,
  workItemTitle: string | null | undefined,
): string | null {
  const name = automationName?.trim().replace(/\s+/g, " ");
  const item = workItemTitle?.trim();
  if (!name || !item) return null;
  return `${name}${AUTOMATION_WORK_ITEM_SEPARATOR}${item}`;
}

/**
 * Resolve a conversation's automation-prefixed work-item name from its tags.
 * Reads the `automationname` tag (the user-configured name of the automation
 * that spawned the conversation) and the work-item snapshot tags. Returns null
 * when either is absent, which is the signal to leave the existing title chain
 * untouched.
 */
export function getAutomationWorkItemTitleFromTags(
  tags: Record<string, string> | null | undefined,
): string | null {
  return composeAutomationWorkItemTitle(
    tags?.[AUTOMATION_NAME_TAG_KEY],
    getWorkItemDisplayTitleFromTags(tags),
  );
}

/**
 * The display-name override for an automation-born, work-item-linked
 * conversation, or null when the existing title chain should stand.
 *
 * The composed name wins the slot only when the server title is *not* an
 * explicit user title — i.e. it is the automation SDK's own machine title
 * (`"{automation name} — {context}"`) or a placeholder. A genuine user rename
 * is neither, so it keeps winning. Placing the override above AI generation
 * matters because an automation run with autotitling still on would otherwise
 * show an unrelated generated title instead of the automation + backlog item.
 */
export function getAutomationWorkItemOverrideTitle(
  tags: Record<string, string> | null | undefined,
  serverTitle: string | null | undefined,
  conversationId: string,
): string | null {
  const composed = getAutomationWorkItemTitleFromTags(tags);
  if (!composed) return null;
  const machine = isAutomationMachineTitle(
    serverTitle,
    tags?.[AUTOMATION_NAME_TAG_KEY],
  );
  return machine || isPlaceholderConversationTitle(serverTitle, conversationId)
    ? composed
    : null;
}
