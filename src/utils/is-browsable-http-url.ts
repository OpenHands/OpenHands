/**
 * Whether a URL may be assigned to a window/popup we control.
 *
 * OAuth authorization URLs arrive from third-party servers (an MCP server's
 * OAuth metadata, a device-flow provider), so they are attacker-controlled
 * input. A popup opened with `window.open("about:blank")` stays on the
 * opener's origin, so navigating it to `javascript:…` runs that script inside
 * the Canvas origin — with access to the session API key in `localStorage`.
 * `data:` and `blob:` URLs are the same class of problem.
 *
 * Only http(s) is browsable: every real authorization endpoint is one, and no
 * other scheme can be navigated to without either executing script in our
 * origin or handing the string to an OS protocol handler.
 */
export function isBrowsableHttpUrl(rawUrl: string): boolean {
  try {
    const { protocol } = new URL(rawUrl);
    return protocol === "http:" || protocol === "https:";
  } catch {
    return false;
  }
}
