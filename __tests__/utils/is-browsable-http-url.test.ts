import { describe, expect, it } from "vitest";

import { isBrowsableHttpUrl } from "#/utils/is-browsable-http-url";

describe("isBrowsableHttpUrl", () => {
  it.each([
    "https://auth.example.com/authorize?client_id=abc",
    "http://localhost:3000/oauth/authorize",
    "http://127.0.0.1:8080/authorize",
  ])("allows the http(s) URL %s", (url) => {
    expect(isBrowsableHttpUrl(url)).toBe(true);
  });

  it.each([
    // Navigating a same-origin popup to these runs script in our origin.
    "javascript:alert(document.cookie)",
    "JavaScript:alert(1)",
    "  javascript:alert(1)",
    "data:text/html,<script>alert(1)</script>",
    "blob:https://example.com/1234",
    // Schemes that would be handed to an OS handler, plus junk.
    "file:///etc/passwd",
    "about:blank",
    "not a url",
    "",
  ])("rejects %s", (url) => {
    expect(isBrowsableHttpUrl(url)).toBe(false);
  });
});
