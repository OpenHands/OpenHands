import { describe, expect, it } from "vitest";

import {
  isExternalBrowsableUrl,
  isLoopbackAppUrl,
} from "./window-url-policy.mjs";

describe("isLoopbackAppUrl", () => {
  it.each([
    "http://localhost:8000/",
    "http://localhost:8000/some/app/path?query=1",
    "http://127.0.0.1:8000/",
    "http://[::1]:8000/",
    "https://localhost:8000/",
  ])("allows the loopback app server: %s", (url) => {
    expect(isLoopbackAppUrl(url)).toBe(true);
  });

  it.each([
    // Prefix look-alikes an attacker controls.
    "http://localhost.evil.com/",
    "http://127.0.0.1.evil.com/",
    "http://localhost@evil.com/",
    "http://127.0.0.1@evil.com/",
    // Remote hosts and non-http schemes.
    "https://openhands.dev/",
    "http://192.168.1.10:8000/",
    "file:///etc/passwd",
    "javascript:alert(1)",
    "about:blank",
    "smb://nas/share",
    "not a url",
    "",
  ])("rejects everything else: %s", (url) => {
    expect(isLoopbackAppUrl(url)).toBe(false);
  });
});

describe("isExternalBrowsableUrl", () => {
  it.each(["https://github.com/openhands", "http://example.com/docs"])(
    "allows browsable schemes: %s",
    (url) => {
      expect(isExternalBrowsableUrl(url)).toBe(true);
    },
  );

  it.each([
    "file:///etc/passwd",
    "file://\\\\nas\\share",
    "smb://nas/share",
    "javascript:alert(1)",
    "about:blank",
    "not a url",
    "",
  ])("denies everything the OS must not handle: %s", (url) => {
    expect(isExternalBrowsableUrl(url)).toBe(false);
  });
});
