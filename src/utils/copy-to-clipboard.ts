/**
 * Copies text to the user's clipboard.
 *
 * First attempts to use the modern async Clipboard API (`navigator.clipboard.writeText`).
 * If unavailable (e.g. non-secure HTTP contexts or older browsers) or if `writeText` rejects,
 * falls back to temporary textarea selection with `document.execCommand("copy")`.
 *
 * Returns `true` if copying succeeded, or `false` if all mechanisms failed.
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  if (
    typeof navigator !== "undefined" &&
    navigator.clipboard &&
    typeof navigator.clipboard.writeText === "function"
  ) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // Fall through to legacy execCommand fallback
    }
  }

  if (
    typeof document !== "undefined" &&
    typeof document.createElement === "function" &&
    typeof document.body?.appendChild === "function"
  ) {
    try {
      const textArea = document.createElement("textarea");
      textArea.value = text;
      textArea.setAttribute("readonly", "");
      textArea.style.contain = "strict";
      textArea.style.position = "fixed";
      textArea.style.left = "-9999px";
      textArea.style.top = "-9999px";
      textArea.style.opacity = "0";

      document.body.appendChild(textArea);

      const selection = document.getSelection();
      const originalRange =
        selection && selection.rangeCount > 0 ? selection.getRangeAt(0) : null;

      let successful = false;
      try {
        textArea.select();
        textArea.setSelectionRange(0, text.length);
        successful = document.execCommand?.("copy") ?? false;
      } finally {
        if (textArea.parentNode) {
          textArea.parentNode.removeChild(textArea);
        }
        if (originalRange && selection) {
          selection.removeAllRanges();
          selection.addRange(originalRange);
        }
      }

      if (successful) {
        return true;
      }
    } catch {
      return false;
    }
  }

  return false;
}
