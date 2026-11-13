/**
 * Copy text to the clipboard, falling back to a selection + execCommand when
 * the async Clipboard API is unavailable.
 *
 * The async API only exists in a secure context and can still reject — a
 * denied permission, or a document that lost focus mid-call. Either way the
 * plain `navigator.clipboard.writeText` this replaces would throw and leave
 * the caller's "copied" state untouched, so the button looked like it did
 * nothing. Returns whether the text actually reached the clipboard.
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // Fall through to the legacy path.
    }
  }

  const active = document.activeElement as HTMLElement | null;
  const selection = document.getSelection();
  const previousRange =
    selection && selection.rangeCount > 0 ? selection.getRangeAt(0) : null;

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  // Keep it out of the layout and out of the tab order without `display:none`,
  // which would make it unselectable.
  textarea.style.position = "fixed";
  textarea.style.top = "-9999px";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);

  try {
    textarea.select();
    return document.execCommand("copy");
  } catch {
    return false;
  } finally {
    document.body.removeChild(textarea);
    if (previousRange && selection) {
      selection.removeAllRanges();
      selection.addRange(previousRange);
    }
    active?.focus();
  }
}
