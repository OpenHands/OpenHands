import { afterEach, describe, expect, it, vi } from "vitest";
import { AxiosError, AxiosHeaders } from "axios";
import { HttpError } from "@openhands/typescript-client";

import { isRetryableError, withRetry } from "#/api/with-retry";

const axiosErrorWithStatus = (status: number) =>
  new AxiosError("Request failed", "ERR_BAD_REQUEST", undefined, undefined, {
    status,
    statusText: "",
    data: {},
    headers: {},
    config: { headers: new AxiosHeaders() },
  });

describe("isRetryableError", () => {
  it.each([400, 401, 403, 404, 409, 422])(
    "does not retry a %i answer",
    (status) => {
      expect(isRetryableError(new HttpError(status, ""))).toBe(false);
      expect(isRetryableError(axiosErrorWithStatus(status))).toBe(false);
    },
  );

  it.each([408, 429, 500, 502, 503])("retries a %i answer", (status) => {
    expect(isRetryableError(new HttpError(status, ""))).toBe(true);
    expect(isRetryableError(axiosErrorWithStatus(status))).toBe(true);
  });

  it("retries failures the server never answered", () => {
    expect(isRetryableError(new TypeError("Failed to fetch"))).toBe(true);
    expect(isRetryableError(new AxiosError("Network Error"))).toBe(true);
    expect(isRetryableError("boom")).toBe(true);
  });
});

describe("withRetry", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("throws a 4xx answer after the first attempt", async () => {
    const error = new HttpError(422, "Unprocessable Entity", {
      detail: "Settings validation failed",
    });
    const fn = vi.fn().mockRejectedValue(error);

    await expect(withRetry(fn)).rejects.toBe(error);
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("retries a 5xx answer with backoff until it succeeds", async () => {
    vi.useFakeTimers();
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new HttpError(503, "Service Unavailable"))
      .mockRejectedValueOnce(new HttpError(503, "Service Unavailable"))
      .mockResolvedValue("ok");

    const result = withRetry(fn);
    await vi.runAllTimersAsync();

    await expect(result).resolves.toBe("ok");
    expect(fn).toHaveBeenCalledTimes(3);
  });

  it("gives up after the last attempt on a retryable error", async () => {
    vi.useFakeTimers();
    const error = new HttpError(500, "Internal Server Error");
    const fn = vi.fn().mockRejectedValue(error);

    const assertion = expect(withRetry(fn)).rejects.toBe(error);
    await vi.runAllTimersAsync();
    await assertion;
    expect(fn).toHaveBeenCalledTimes(3);
  });
});
