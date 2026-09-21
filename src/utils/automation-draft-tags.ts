export const AUTOMATION_SETUP_TAG_KEY = "automationsetup";
export const AUTOMATION_DRAFT_ID_TAG_KEY = "automationdraftid";
export const AUTOMATION_SETUP_TAG_VALUE = "draft";

export function getAutomationDraftIdFromTags(
  tags: Record<string, string> | null | undefined,
): string | null {
  const draftId = tags?.[AUTOMATION_DRAFT_ID_TAG_KEY]?.trim();
  return draftId || null;
}

export function buildAutomationDraftTags(
  tags: Record<string, string> | null | undefined,
  draftId: string,
): Record<string, string> {
  return {
    ...(tags ?? {}),
    [AUTOMATION_SETUP_TAG_KEY]: AUTOMATION_SETUP_TAG_VALUE,
    [AUTOMATION_DRAFT_ID_TAG_KEY]: draftId,
  };
}

export function removeAutomationDraftTags(
  tags: Record<string, string> | null | undefined,
): Record<string, string> {
  const next = { ...(tags ?? {}) };
  delete next[AUTOMATION_SETUP_TAG_KEY];
  delete next[AUTOMATION_DRAFT_ID_TAG_KEY];
  return next;
}
