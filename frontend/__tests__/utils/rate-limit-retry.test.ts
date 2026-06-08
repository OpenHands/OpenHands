import { AxiosError } from "axios";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  getRateLimitRetryDelayMs,
  isRateLimitError,
} from "#/utils/rate-limit-retry";

const createAxiosError = (status: number, headers?: unknown): AxiosError =>
  ({
    response: {
      status,
      headers,
    },
  }) as unknown as AxiosError;

describe("rate limit retry helpers", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("recognizes response status 429 as a rate-limit error", () => {
    expect(isRateLimitError(createAxiosError(429))).toBe(true);
    expect(isRateLimitError(createAxiosError(500))).toBe(false);
  });

  it("uses AxiosHeaders get() before falling back to enumerable headers", () => {
    const headers = {
      "retry-after": "5",
      get: (headerName: string) =>
        headerName.toLowerCase() === "retry-after" ? "2" : undefined,
    };

    expect(getRateLimitRetryDelayMs(createAxiosError(429, headers))).toBe(2000);
  });

  it("parses Retry-After HTTP dates", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-06-04T20:00:00.000Z"));

    const retryAt = new Date("2026-06-04T20:00:03.000Z").toUTCString();

    expect(
      getRateLimitRetryDelayMs(
        createAxiosError(429, {
          "retry-after": retryAt,
        }),
      ),
    ).toBe(3000);
  });

  it("falls back to one second when Retry-After is missing", () => {
    expect(getRateLimitRetryDelayMs(createAxiosError(429))).toBe(1000);
  });

  it("falls back to one second for non-numeric Retry-After", () => {
    expect(
      getRateLimitRetryDelayMs(
        createAxiosError(429, { "retry-after": "invalid" }),
      ),
    ).toBe(1000);
  });

  it("treats negative Retry-After seconds as immediate retry (already expired)", () => {
    // Date.parse("-5") returns a valid timestamp (epoch - 5000ms), and since
    // that time has passed, Math.max(negative - now, 0) returns 0
    expect(
      getRateLimitRetryDelayMs(createAxiosError(429, { "retry-after": "-5" })),
    ).toBe(0);
  });

  it("treats zero Retry-After seconds as immediate (0ms delay)", () => {
    expect(
      getRateLimitRetryDelayMs(createAxiosError(429, { "retry-after": "0" })),
    ).toBe(0);
  });

  it("clamps large Retry-After seconds to max 60 seconds", () => {
    expect(
      getRateLimitRetryDelayMs(createAxiosError(429, { "retry-after": "120" })),
    ).toBe(60_000);
  });

  it("clamps far-future Retry-After dates to max 60 seconds", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-06-04T20:00:00.000Z"));

    const farFuture = new Date("2026-06-04T21:00:00.000Z").toUTCString();

    expect(
      getRateLimitRetryDelayMs(createAxiosError(429, { "retry-after": farFuture })),
    ).toBe(60_000);
  });
});
