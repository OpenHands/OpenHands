import { AxiosError } from "axios";
import { HttpError } from "@openhands/typescript-client";
import { describe, expect, it } from "vitest";
import { retryTransientRequest } from "#/utils/transient-error-retry";

describe("transient request retry", () => {
  it.each([408, 429, 500, 502, 503, 504])(
    "retries HTTP %i with a finite budget",
    (status) => {
      const error = new HttpError(status, "Unavailable", null);
      expect(retryTransientRequest(0, error)).toBe(true);
      expect(retryTransientRequest(2, error)).toBe(true);
      expect(retryTransientRequest(3, error)).toBe(false);
    },
  );
  it.each([400, 401, 403, 404, 409, 422, 501])(
    "does not retry HTTP %i",
    (status) => {
      expect(
        retryTransientRequest(0, new HttpError(status, "Failure", null)),
      ).toBe(false);
    },
  );
  it.each([
    new TypeError("Failed to fetch"),
    new AxiosError("offline", "ERR_NETWORK"),
    new AxiosError("timeout", "ECONNABORTED"),
    new AxiosError("unavailable", undefined, undefined, undefined, {
      status: 503,
    } as never),
  ])("retries transport failures", (error) => {
    expect(retryTransientRequest(0, error)).toBe(true);
  });
  it("does not retry an unclassified application failure or a cancellation", () => {
    expect(retryTransientRequest(0, new Error("invalid response"))).toBe(false);
    expect(
      retryTransientRequest(0, new AxiosError("canceled", "ERR_CANCELED")),
    ).toBe(false);
  });
});
