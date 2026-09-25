export const AUTOMATION_SETUP_TAG_KEY = "automationsetup";
export const AUTOMATION_DRAFT_ID_TAG_KEY = "automationdraftid";
export const AUTOMATION_MATERIALIZED_DRAFT_ID_TAG_KEY =
  "automationmaterializeddraftid";
export const AUTOMATION_SETUP_TAG_VALUE = "draft";

export function getAutomationDraftIdFromTags(
  tags: Record<string, string> | null | undefined,
): string | null {
  const draftId = tags?.[AUTOMATION_DRAFT_ID_TAG_KEY]?.trim();
  return draftId || null;
}

export function getAutomationMaterializedDraftIdFromTags(
  tags: Record<string, string> | null | undefined,
): string | null {
  const draftId = tags?.[AUTOMATION_MATERIALIZED_DRAFT_ID_TAG_KEY]?.trim();
  return draftId || null;
}

export function hasAutomationSetupModeTag(
  tags: Record<string, string> | null | undefined,
): boolean {
  return tags?.[AUTOMATION_SETUP_TAG_KEY] === AUTOMATION_SETUP_TAG_VALUE;
}

export function buildAutomationSetupModeTags(
  tags: Record<string, string> | null | undefined,
): Record<string, string> {
  return {
    ...(tags ?? {}),
    [AUTOMATION_SETUP_TAG_KEY]: AUTOMATION_SETUP_TAG_VALUE,
  };
}

export function buildAutomationDraftTags(
  tags: Record<string, string> | null | undefined,
  draftId: string,
  materializedDraftId?: string | null,
): Record<string, string> {
  const next: Record<string, string> = {
    ...buildAutomationSetupModeTags(tags),
    [AUTOMATION_DRAFT_ID_TAG_KEY]: draftId,
  };
  if (materializedDraftId) {
    next[AUTOMATION_MATERIALIZED_DRAFT_ID_TAG_KEY] = materializedDraftId;
  }
  return next;
}

export function removeAutomationDraftTags(
  tags: Record<string, string> | null | undefined,
): Record<string, string> {
  const next = { ...(tags ?? {}) };
  delete next[AUTOMATION_SETUP_TAG_KEY];
  delete next[AUTOMATION_DRAFT_ID_TAG_KEY];
  delete next[AUTOMATION_MATERIALIZED_DRAFT_ID_TAG_KEY];
  return next;
}
