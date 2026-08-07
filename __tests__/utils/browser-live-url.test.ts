import { describe, expect, it } from "vitest";

import {
  isValidLivePreviewUrl,
  normalizeLivePreviewUrl,
} from "#/utils/browser-live-url";

describe("browser-live-url", () => {
  it.each([
    ["https://example.com", true],
    ["http://localhost:8089", true],
    ["javascript:alert(1)", false],
    ["data:text/html,hi", false],
    ["", false],
    [null, false],
  ])("isValidLivePreviewUrl(%j) → %s", (value, expected) => {
    expect(isValidLivePreviewUrl(value)).toBe(expected);
  });

  it.each([
    ["localhost:8089", "http://localhost:8089"],
    ["127.0.0.1:3000/path", "http://127.0.0.1:3000/path"],
    ["example.com", "https://example.com"],
    ["https://already.https", "https://already.https"],
    ["  http://spaced.test  ", "http://spaced.test"],
    ["javascript:alert(1)", null],
    ["", null],
  ])("normalizeLivePreviewUrl(%j) → %j", (raw, expected) => {
    expect(normalizeLivePreviewUrl(raw)).toBe(expected);
  });
});
