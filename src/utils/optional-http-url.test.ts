import { describe, expect, it } from "vitest";
import { isValidOptionalHttpUrl } from "./optional-http-url";

describe("isValidOptionalHttpUrl", () => {
  it.each([
    ["", true],
    ["   ", true],
    ["https://api.openai.com", true],
    ["https://api.openai.com ", true],
    ["  https://api.openai.com/v1  ", true],
    ["http://localhost:11434", true],
    ["http://127.0.0.1:8000", true],
    ["not-a-url", false],
    [".", false],
    ["api.openai.com", false],
    ["localhost:11434", false],
    ["ftp://example.com", false],
    ["https://", false],
    ["http://", false],
  ])("treats %j as valid=%s", (value, expected) => {
    expect(isValidOptionalHttpUrl(value)).toBe(expected);
  });
});
