import { describe, expect, it } from "vitest";
import { shouldRetrySettingsQuery } from "./use-settings";

describe("shouldRetrySettingsQuery", () => {
  it("does not retry a missing settings response", () => {
    expect(shouldRetrySettingsQuery(0, { status: 404 })).toBe(false);
  });

  it("retries transient failures at most three times", () => {
    const transientError = { status: 500 };

    expect(shouldRetrySettingsQuery(0, transientError)).toBe(true);
    expect(shouldRetrySettingsQuery(1, transientError)).toBe(true);
    expect(shouldRetrySettingsQuery(2, transientError)).toBe(true);
    expect(shouldRetrySettingsQuery(3, transientError)).toBe(false);
  });
});
