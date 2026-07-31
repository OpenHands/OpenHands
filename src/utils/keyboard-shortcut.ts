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

/**
 * True when the event is the command-palette chord (⌘/Ctrl + key).
 * Accepts either meta or ctrl so Mac and Windows/Linux both work, matching
 * the global command-menu shortcut behavior.
 */
export function matchesPrimaryModifierShortcut(
  event: KeyboardEvent,
  key: string,
): boolean {
  return (
    event.key.toLowerCase() === key.toLowerCase() &&
    (event.metaKey || event.ctrlKey) &&
    !event.altKey &&
    !event.shiftKey
  );
}
