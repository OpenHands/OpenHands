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
    vi.restoreAllMocks();
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

    vi.spyOn(Math, "random").mockReturnValue(0);

    expect(getRateLimitRetryDelayMs(0, createAxiosError(429, headers))).toBe(
      2000,
    );
  });

  it("accepts numeric Retry-After values from plain headers", () => {
    vi.spyOn(Math, "random").mockReturnValue(0);

    expect(
      getRateLimitRetryDelayMs(
        0,
        createAxiosError(429, { "Retry-After": 2 }),
      ),
    ).toBe(2000);
  });

  it("falls back to backoff when AxiosHeaders returns a non-string value", () => {
    vi.spyOn(Math, "random").mockReturnValue(0);

    expect(
      getRateLimitRetryDelayMs(
        0,
        createAxiosError(429, { get: () => 2 }),
      ),
    ).toBe(1000);
  });

  it("falls back to backoff for unsupported plain header values", () => {
    vi.spyOn(Math, "random").mockReturnValue(0);

    expect(
      getRateLimitRetryDelayMs(
        0,
        createAxiosError(429, { "retry-after": { seconds: 2 } }),
      ),
    ).toBe(1000);
  });

  it("parses Retry-After HTTP dates", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-06-04T20:00:00.000Z"));

    const retryAt = new Date("2026-06-04T20:00:03.000Z").toUTCString();
    vi.spyOn(Math, "random").mockReturnValue(0);

    expect(
      getRateLimitRetryDelayMs(
        0,
        createAxiosError(429, {
          "retry-after": retryAt,
        }),
      ),
    ).toBe(3000);
  });

  it("falls back to one second when Retry-After is missing", () => {
    vi.spyOn(Math, "random").mockReturnValue(0);

    expect(getRateLimitRetryDelayMs(0, createAxiosError(429))).toBe(1000);
  });

  it("falls back to one second for non-numeric Retry-After", () => {
    vi.spyOn(Math, "random").mockReturnValue(0);

    expect(
      getRateLimitRetryDelayMs(
        0,
        createAxiosError(429, { "retry-after": "invalid" }),
      ),
    ).toBe(1000);
  });

  it("uses the initial backoff when Retry-After is already expired", () => {
    vi.spyOn(Math, "random").mockReturnValue(0);

    expect(
      getRateLimitRetryDelayMs(
        0,
        createAxiosError(429, { "retry-after": "-5" }),
      ),
    ).toBe(1000);
  });

  it("uses the initial backoff when Retry-After is zero", () => {
    vi.spyOn(Math, "random").mockReturnValue(0);

    expect(
      getRateLimitRetryDelayMs(
        0,
        createAxiosError(429, { "retry-after": "0" }),
      ),
    ).toBe(1000);
  });

  it("clamps large Retry-After seconds to max 60 seconds", () => {
    vi.spyOn(Math, "random").mockReturnValue(0);

    expect(
      getRateLimitRetryDelayMs(
        0,
        createAxiosError(429, { "retry-after": "120" }),
      ),
    ).toBe(60_000);
  });

  it("clamps far-future Retry-After dates to max 60 seconds", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-06-04T20:00:00.000Z"));

    const farFuture = new Date("2026-06-04T21:00:00.000Z").toUTCString();
    vi.spyOn(Math, "random").mockReturnValue(0);

    expect(
      getRateLimitRetryDelayMs(
        0,
        createAxiosError(429, { "retry-after": farFuture }),
      ),
    ).toBe(60_000);
  });

  it("grows the fallback delay exponentially for repeated failures", () => {
    vi.spyOn(Math, "random").mockReturnValue(0);

    expect(getRateLimitRetryDelayMs(0, createAxiosError(429))).toBe(1000);
    expect(getRateLimitRetryDelayMs(1, createAxiosError(429))).toBe(2000);
    expect(getRateLimitRetryDelayMs(2, createAxiosError(429))).toBe(4000);
  });

  it("adds jitter while never retrying before Retry-After", () => {
    vi.spyOn(Math, "random").mockReturnValue(0.5);

    expect(getRateLimitRetryDelayMs(0, createAxiosError(429))).toBe(1500);
    // jitter scales off the 5s Retry-After: 0.5 * min(5000, MAX_JITTER_MS) = 2500
    expect(
      getRateLimitRetryDelayMs(
        0,
        createAxiosError(429, { "retry-after": "5" }),
      ),
    ).toBe(7500);
  });

  it("caps jitter to avoid unbounded delays", () => {
    vi.spyOn(Math, "random").mockReturnValue(0.999);

    expect(getRateLimitRetryDelayMs(10, createAxiosError(429))).toBe(64_995);
  });
});
