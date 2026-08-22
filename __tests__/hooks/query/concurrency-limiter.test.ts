import { describe, expect, it, vi } from "vitest";
import {
  ConcurrencyLimiter,
  automationRunRequestsLimiter,
} from "#/hooks/query/concurrency-limiter";

describe("ConcurrencyLimiter", () => {
  it("runs a task immediately when a slot is available", async () => {
    const limiter = new ConcurrencyLimiter(2);
    const result = await limiter.run(async () => "done");
    expect(result).toBe("done");
  });

  it("does not exceed the configured maximum concurrency", async () => {
    const limiter = new ConcurrencyLimiter(2);
    let running = 0;
    let maxObserved = 0;

    const makeTask = async (id: number) => {
      running += 1;
      maxObserved = Math.max(maxObserved, running);
      await new Promise((resolve) => setTimeout(resolve, 20));
      running -= 1;
      return id;
    };

    const ids = await Promise.all(
      Array.from({ length: 5 }, (_, i) => limiter.run(() => makeTask(i))),
    );

    expect(ids).toEqual([0, 1, 2, 3, 4]);
    expect(maxObserved).toBe(2);
    expect(running).toBe(0);
  });

  it("propagates task errors without leaking the active slot", async () => {
    const limiter = new ConcurrencyLimiter(1);
    await expect(
      limiter.run(async () => {
        throw new Error("boom");
      }),
    ).rejects.toThrow("boom");

    // A subsequent task should still be able to run.
    const result = await limiter.run(async () => "ok");
    expect(result).toBe("ok");
  });

  it("queues tasks and keeps order under the shared cap", async () => {
    const limiter = new ConcurrencyLimiter(1);
    const order: number[] = [];

    const task = (id: number) =>
      limiter.run(async () => {
        order.push(id);
        return id;
      });

    await Promise.all(Array.from({ length: 4 }, (_, i) => task(i)));

    expect(order).toEqual([0, 1, 2, 3]);
  });

  it("shares the same automation-run limiter across tests", () => {
    // The exported limiter is a singleton so multiple fan-out sites cannot
    // accidentally create their own limiters and bypass the global cap.
    expect(automationRunRequestsLimiter).toBeInstanceOf(ConcurrencyLimiter);
  });
});
