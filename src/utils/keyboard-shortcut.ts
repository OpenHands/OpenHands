export function isMacPlatform(): boolean {
  if (typeof navigator === "undefined") {
    return false;
  }
  return /Mac|iPhone|iPad|iPod/.test(navigator.platform);
}

export function formatPrimaryModifierShortcut(key: string): string {
  const normalizedKey = key.length === 1 ? key.toUpperCase() : key;
  return isMacPlatform() ? `⌘${normalizedKey}` : `Ctrl+${normalizedKey}`;
}

export function matchesPrimaryModifierShortcut(
  event: KeyboardEvent,
  key: string,
): boolean {
  const usesMeta = isMacPlatform();
  return (
    event.key.toLowerCase() === key.toLowerCase() &&
    (usesMeta ? event.metaKey : event.ctrlKey) &&
    !event.altKey &&
    !event.shiftKey
  );
}

export function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }

  const tagName = target.tagName;
  if (tagName === "INPUT" || tagName === "TEXTAREA" || tagName === "SELECT") {
    return true;
  }

  return target.isContentEditable === true;
}
