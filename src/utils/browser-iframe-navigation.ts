/**
 * Helpers for driving the live-preview iframe's browsing context from the
 * parent page. Cross-origin frames allow history.back/forward and
 * location.reload in Chromium without reading sensitive location data;
 * reading location.href still requires same-origin (used to sync the
 * address bar after in-iframe link clicks).
 */

export function tryIframeGoBack(iframe: HTMLIFrameElement | null): boolean {
  try {
    iframe?.contentWindow?.history.back();
    return Boolean(iframe?.contentWindow);
  } catch {
    return false;
  }
}

export function tryIframeGoForward(iframe: HTMLIFrameElement | null): boolean {
  try {
    iframe?.contentWindow?.history.forward();
    return Boolean(iframe?.contentWindow);
  } catch {
    return false;
  }
}

export function tryIframeReload(iframe: HTMLIFrameElement | null): boolean {
  try {
    iframe?.contentWindow?.location.reload();
    return Boolean(iframe?.contentWindow);
  } catch {
    return false;
  }
}

export function tryReadIframeHref(
  iframe: HTMLIFrameElement | null,
): string | null {
  try {
    const href = iframe?.contentWindow?.location.href;
    if (!href || href === "about:blank") {
      return null;
    }
    return href;
  } catch {
    return null;
  }
}
