import { describe, expect, it, vi } from "vitest";
import { withRetry } from "./with-retry";

describe("withRetry", () => {
  it("returns the first success without retrying", async () => {
    const fn = vi.fn().mockResolvedValue("ok");
    await expect(withRetry(fn, 3, 1)).resolves.toBe("ok");
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("retries every failure by default and returns a later success", async () => {
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new Error("boom"))
      .mockResolvedValue("recovered");
    await expect(withRetry(fn, 3, 1)).resolves.toBe("recovered");
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it("throws the last error after exhausting attempts", async () => {
    const fn = vi.fn().mockRejectedValue(new Error("down"));
    await expect(withRetry(fn, 2, 1)).rejects.toThrow("down");
    expect(fn).toHaveBeenCalledTimes(2);
  });

  it("does not retry when shouldRetry returns false", async () => {
    const fn = vi.fn().mockRejectedValue(new Error("client error"));
    await expect(withRetry(fn, 3, 1, () => false)).rejects.toThrow(
      "client error",
    );
    expect(fn).toHaveBeenCalledTimes(1);
  });

  it("stops retrying as soon as a failure is not retryable", async () => {
    const fn = vi
      .fn()
      .mockRejectedValueOnce(new Error("transient"))
      .mockRejectedValue(new Error("fatal"));
    const shouldRetry = (error: unknown) =>
      (error as Error).message !== "fatal";
    await expect(withRetry(fn, 3, 1, shouldRetry)).rejects.toThrow("fatal");
    expect(fn).toHaveBeenCalledTimes(2);
  });
});
