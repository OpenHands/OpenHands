export type AutomationSetupKind = "prompt" | "plugin" | "custom";

export interface AutomationSetupDraft {
  prompt: string;
  kind: AutomationSetupKind;
  plugins?: string[];
}

const AUTOMATION_SETUP_DRAFTS_STORAGE_KEY = "openhands-automation-setup-drafts";

function readAllDrafts(): Record<string, AutomationSetupDraft> {
  if (typeof window === "undefined") return {};

  try {
    const raw = window.sessionStorage.getItem(
      AUTOMATION_SETUP_DRAFTS_STORAGE_KEY,
    );
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    if (
      typeof parsed !== "object" ||
      parsed === null ||
      Array.isArray(parsed)
    ) {
      return {};
    }
    return parsed as Record<string, AutomationSetupDraft>;
  } catch {
    return {};
  }
}

function writeAllDrafts(drafts: Record<string, AutomationSetupDraft>) {
  if (typeof window === "undefined") return;

  try {
    window.sessionStorage.setItem(
      AUTOMATION_SETUP_DRAFTS_STORAGE_KEY,
      JSON.stringify(drafts),
    );
  } catch {
    // sessionStorage not available
  }
}

function normalizeDraft(value: AutomationSetupDraft): AutomationSetupDraft {
  return {
    prompt: value.prompt,
    kind: value.kind,
    ...(value.plugins && value.plugins.length > 0
      ? { plugins: [...value.plugins] }
      : {}),
  };
}

export function getAutomationSetupDraft(
  conversationId: string | null | undefined,
): AutomationSetupDraft | null {
  if (!conversationId) return null;
  const draft = readAllDrafts()[conversationId];
  if (!draft || typeof draft.prompt !== "string") return null;
  if (!["prompt", "plugin", "custom"].includes(draft.kind)) return null;
  return normalizeDraft(draft);
}

export function setAutomationSetupDraft(
  conversationId: string,
  draft: AutomationSetupDraft,
) {
  writeAllDrafts({
    ...readAllDrafts(),
    [conversationId]: normalizeDraft(draft),
  });
}

export function clearAutomationSetupDraft(conversationId: string) {
  const drafts = readAllDrafts();
  delete drafts[conversationId];
  writeAllDrafts(drafts);
}
