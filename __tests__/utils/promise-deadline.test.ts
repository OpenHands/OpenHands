import { afterEach, describe, expect, it, vi } from "vitest";
import {
  PromiseDeadlineError,
  withPromiseDeadline,
} from "#/utils/promise-deadline";

describe("withPromiseDeadline", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  it("clears its timer when the operation settles first", async () => {
    vi.useFakeTimers();

    await expect(
      withPromiseDeadline(Promise.resolve("done"), 1000, "late"),
    ).resolves.toBe("done");

    expect(vi.getTimerCount()).toBe(0);
  });

  it("rejects at the exact deadline and ignores late settlement", async () => {
    vi.useFakeTimers();
    let rejectOperation!: (error: Error) => void;
    const operation = new Promise<never>((_resolve, reject) => {
      rejectOperation = reject;
    });
    const bounded = withPromiseDeadline(operation, 1000, "deadline reached");
    const observed = bounded.catch((error: unknown) => error);

    await vi.advanceTimersByTimeAsync(999);
    expect(vi.getTimerCount()).toBe(1);
    await vi.advanceTimersByTimeAsync(1);

    await expect(observed).resolves.toEqual(
      expect.objectContaining<Partial<PromiseDeadlineError>>({
        name: "PromiseDeadlineError",
        message: "deadline reached",
      }),
    );
    rejectOperation(new Error("late rejection"));
    await Promise.resolve();
  });
});
