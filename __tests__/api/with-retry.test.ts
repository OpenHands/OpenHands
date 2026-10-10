import { describe, expect, it, vi } from "vitest";
import { withRetry } from "#/api/with-retry";

describe("withRetry", () => {
  it("does not retry non-retryable HTTP errors", async () => {
    const error = Object.assign(new Error("not found"), { status: 404 });
    const fn = vi.fn().mockRejectedValue(error);

    await expect(withRetry(fn, 3, 0)).rejects.toBe(error);

    expect(fn).toHaveBeenCalledOnce();
  });

  it("retries retryable HTTP errors", async () => {
    const error = { response: { status: 503 } };
    const fn = vi.fn().mockRejectedValue(error);

    await expect(withRetry(fn, 3, 0)).rejects.toBe(error);

    expect(fn).toHaveBeenCalledTimes(3);
  });

  it.each([
    Object.assign(new Error("aborted"), { name: "AbortError" }),
    Object.assign(new Error("cancelled"), { code: "ERR_CANCELED" }),
    new Error("wrapped", {
      cause: Object.assign(new Error("aborted"), { name: "AbortError" }),
    }),
  ])("does not retry cancellation errors", async (error) => {
    const fn = vi.fn().mockRejectedValue(error);

    await expect(withRetry(fn, 3, 0)).rejects.toBe(error);

    expect(fn).toHaveBeenCalledOnce();
  });
});
