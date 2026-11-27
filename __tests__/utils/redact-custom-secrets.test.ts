import { describe, it, expect } from "vitest";
import { redactCustomSecrets } from "#/utils/redact-custom-secrets";

describe("redactCustomSecrets", () => {
  it("preserves already-masked secret values", () => {
    const text =
      "<CUSTOM_SECRETS>\nMY_API_KEY=<secret-hidden>\n</CUSTOM_SECRETS>";
    expect(redactCustomSecrets(text)).toBe(text);
  });

  it("redacts an unmasked value inside the block (= separator)", () => {
    const text = "<CUSTOM_SECRETS>\nMY_API_KEY=leaked-value\n</CUSTOM_SECRETS>";
    const result = redactCustomSecrets(text);
    expect(result).not.toContain("leaked-value");
    expect(result).toContain("MY_API_KEY=<secret-hidden>");
  });

  it("redacts an unmasked value inside the block (: separator)", () => {
    const text =
      "<CUSTOM_SECRETS>\nMY_API_KEY: leaked-value\n</CUSTOM_SECRETS>";
    const result = redactCustomSecrets(text);
    expect(result).not.toContain("leaked-value");
    expect(result).toContain("MY_API_KEY: <secret-hidden>");
  });

  it("redacts only inside the block, leaving surrounding text untouched", () => {
    const text =
      "<CURRENT_DATETIME>2026-06-01</CURRENT_DATETIME>\n" +
      "<CUSTOM_SECRETS>\nTOKEN=abc123\n</CUSTOM_SECRETS>\n" +
      "key=value-outside-block";
    const result = redactCustomSecrets(text);
    expect(result).toContain("<CURRENT_DATETIME>2026-06-01</CURRENT_DATETIME>");
    expect(result).toContain("key=value-outside-block");
    expect(result).not.toContain("abc123");
    expect(result).toContain("TOKEN=<secret-hidden>");
  });

  it("returns text unchanged when there is no CUSTOM_SECRETS block", () => {
    const text = "<SKILLS>my-skill</SKILLS>\nkey=value";
    expect(redactCustomSecrets(text)).toBe(text);
  });

  it("redacts multiple secrets within the block", () => {
    const text =
      "<CUSTOM_SECRETS>\nA=one\nB=<secret-hidden>\nC=three\n</CUSTOM_SECRETS>";
    const result = redactCustomSecrets(text);
    expect(result).not.toContain("one");
    expect(result).not.toContain("three");
    expect(result).toContain("A=<secret-hidden>");
    expect(result).toContain("B=<secret-hidden>");
    expect(result).toContain("C=<secret-hidden>");
  });

  it("redacts even when the closing tag is missing (truncated block)", () => {
    const text = "<CUSTOM_SECRETS>\nMY_API_KEY=leaked-value";
    const result = redactCustomSecrets(text);
    expect(result).not.toContain("leaked-value");
    expect(result).toContain("MY_API_KEY=<secret-hidden>");
  });

  it("stays idempotent across repeated calls", () => {
    const text = "<CUSTOM_SECRETS>\nTOKEN=leaked-value\n</CUSTOM_SECRETS>";
    const once = redactCustomSecrets(text);
    const twice = redactCustomSecrets(once);
    expect(twice).toBe(once);
    expect(twice).not.toContain("leaked-value");
  });
});
