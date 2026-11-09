import { AxiosError } from "axios";
import { afterEach, describe, expect, it, vi } from "vitest";
import { getQueryRetryDelay, isRateLimitError } from "#/utils/rate-limit-retry";

afterEach(() => {
  vi.restoreAllMocks();
});

describe("isRateLimitError", () => {
  it("recognizes a 429 surfaced via AxiosError.response.status", () => {
    const error = new AxiosError(
      "Too Many Requests",
      "ERR_BAD_REQUEST",
      undefined,
      undefined,
      {
        status: 429,
      } as never,
    );
    expect(isRateLimitError(error)).toBe(true);
  });

  it("recognizes a 429 surfaced via AxiosError.status directly", () => {
    const error = new AxiosError("Too Many Requests");
    error.status = 429;
    expect(isRateLimitError(error)).toBe(true);
  });

  it("recognizes a 429 on a duck-typed HttpError-shaped object", () => {
    expect(isRateLimitError({ status: 429 })).toBe(true);
  });

  it("returns false for non-429 errors", () => {
    expect(isRateLimitError(new AxiosError("Server error"))).toBe(false);
    expect(isRateLimitError({ status: 500 })).toBe(false);
    expect(isRateLimitError(null)).toBe(false);
    expect(isRateLimitError(undefined)).toBe(false);
    expect(isRateLimitError("not an error")).toBe(false);
  });
});

describe("getQueryRetryDelay", () => {
  it("caps the delay and never returns a negative or zero value", () => {
    for (let attempt = 0; attempt < 10; attempt += 1) {
      const delay = getQueryRetryDelay(attempt, new Error("boom"));
      expect(delay).toBeGreaterThan(0);
      expect(delay).toBeLessThanOrEqual(30000);
    }
  });

  it("backs off further for a confirmed rate-limit error than a generic error at the same attempt", () => {
    vi.spyOn(Math, "random").mockReturnValue(1);
    const rateLimited = getQueryRetryDelay(1, { status: 429 });
    const generic = getQueryRetryDelay(1, new Error("boom"));
    expect(rateLimited).toBeGreaterThan(generic);
  });

  it("applies jitter so repeated calls for the same attempt are not identical", () => {
    const delays = new Set(
      Array.from({ length: 20 }, () => getQueryRetryDelay(3, { status: 429 })),
    );
    expect(delays.size).toBeGreaterThan(1);
  });
});
