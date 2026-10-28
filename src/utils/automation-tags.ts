// The automation service's tag rules, mirrored so the editor can reject a tag
// before the PATCH does and save exactly what it shows.
export const AUTOMATION_TAG_PATTERN = /^[a-z0-9]+(-[a-z0-9]+)*$/;
export const AUTOMATION_TAG_MAX_LENGTH = 50;
export const AUTOMATION_TAG_MAX_COUNT = 10;

export type AutomationTagError = "invalid" | "too-long" | "too-many";

/** "  Software  Factory " -> "software-factory", as the service stores it. */
export function normalizeAutomationTag(raw: string): string {
  return raw.trim().toLowerCase().replace(/\s+/g, "-");
}

/** Why `tag` cannot join `tags`, or null when it can (a repeat is a no-op). */
export function getAutomationTagError(
  tags: readonly string[],
  tag: string,
): AutomationTagError | null {
  if (tag.length > AUTOMATION_TAG_MAX_LENGTH) return "too-long";
  if (!AUTOMATION_TAG_PATTERN.test(tag)) return "invalid";
  if (!tags.includes(tag) && tags.length >= AUTOMATION_TAG_MAX_COUNT) {
    return "too-many";
  }
  return null;
}
