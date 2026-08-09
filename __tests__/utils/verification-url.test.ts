import { describe, it, expect } from "vitest";
import { isValidVerificationUrl } from "#/utils/verification-url";

describe("isValidVerificationUrl", () => {
  it("accepts https URLs", () => {
    expect(isValidVerificationUrl("https://auth.openai.com/device")).toBe(true);
  });

  it("rejects non-https schemes", () => {
    expect(isValidVerificationUrl("http://auth.openai.com/device")).toBe(false);
    expect(isValidVerificationUrl("javascript:alert(1)")).toBe(false);
    expect(isValidVerificationUrl("data:text/html,<script>alert(1)</script>")).toBe(
      false,
    );
    expect(isValidVerificationUrl("not-a-url")).toBe(false);
  });
});
