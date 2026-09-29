export const AUTOMATION_SETUP_TAG_KEY = "automationsetup";
export const AUTOMATION_SETUP_TAG_VALUE = "draft";

export function buildAutomationSetupModeTags(
  tags: Record<string, string> | null | undefined,
): Record<string, string> {
  return {
    ...(tags ?? {}),
    [AUTOMATION_SETUP_TAG_KEY]: AUTOMATION_SETUP_TAG_VALUE,
  };
}
