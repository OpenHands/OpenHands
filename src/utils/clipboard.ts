/**
 * Copy text to the clipboard, with a fallback for non-secure contexts.
 *
 * `navigator.clipboard` is only exposed in secure contexts (HTTPS, localhost,
 * 127.0.0.1, file:). When the frontend is served over plain HTTP on a
 * non-loopback host (e.g. a remote IPv6 address), `navigator.clipboard` is
 * `undefined` and `navigator.clipboard.writeText` throws. This helper falls
 * back to the deprecated-but-widely-supported `document.execCommand("copy")`
 * via a transient textarea so copy still works in those contexts.
 *
 * Returns true on success, false on failure.
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // Fall through to the execCommand fallback.
    }
  }
  return copyWithExecCommand(text);
}

function copyWithExecCommand(text: string): boolean {
  if (typeof document === "undefined" || !document.execCommand) {
    return false;
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  // Move off-screen and hide so the fallback is invisible; `display: none`
  // prevents selection, so use positioning + opacity instead.
  textarea.style.position = "fixed";
  textarea.style.top = "-9999px";
  textarea.style.left = "-9999px";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();

  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch {
    ok = false;
  }

  document.body.removeChild(textarea);
  return ok;
}
