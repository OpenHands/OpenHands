const AUTOMATION_SETUP_HANDOFFS_STORAGE_KEY =
  "openhands-automation-setup-handoffs";

function readHandoffs(): Record<string, true> {
  if (typeof window === "undefined") return {};

  try {
    const raw = window.sessionStorage.getItem(
      AUTOMATION_SETUP_HANDOFFS_STORAGE_KEY,
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
    return Object.fromEntries(
      Object.entries(parsed).filter(([, value]) => value === true),
    ) as Record<string, true>;
  } catch {
    return {};
  }
}

function writeHandoffs(handoffs: Record<string, true>) {
  if (typeof window === "undefined") return;

  try {
    window.sessionStorage.setItem(
      AUTOMATION_SETUP_HANDOFFS_STORAGE_KEY,
      JSON.stringify(handoffs),
    );
  } catch {
    // sessionStorage not available
  }
}

export function markAutomationSetupHandoff(conversationId: string) {
  writeHandoffs({
    ...readHandoffs(),
    [conversationId]: true,
  });
}

export function consumeAutomationSetupHandoff(
  conversationId: string | null | undefined,
): boolean {
  if (!conversationId) return false;
  const handoffs = readHandoffs();
  const hasHandoff = handoffs[conversationId] === true;
  if (hasHandoff) {
    delete handoffs[conversationId];
    writeHandoffs(handoffs);
  }
  return hasHandoff;
}

export function clearAutomationSetupHandoff(conversationId: string) {
  const handoffs = readHandoffs();
  delete handoffs[conversationId];
  writeHandoffs(handoffs);
}
