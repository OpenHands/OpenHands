import { describe, it, expect } from "vitest";
import { ConcurrencyLimiter } from "#/hooks/query/concurrency-limiter";

describe("ConcurrencyLimiter", () => {
  it("runs a single task immediately", async () => {
    const limiter = new ConcurrencyLimiter(3);
    const result = await limiter.run(async () => "done");
    expect(result).toBe("done");
  });

  it("limits concurrent tasks to max", async () => {
    const limiter = new ConcurrencyLimiter(2);

    let concurrent = 0;
    let maxConcurrent = 0;

    const task = async () => {
      concurrent++;
      maxConcurrent = Math.max(maxConcurrent, concurrent);
      // Simulate async work without real timing
      await new Promise((r) => setTimeout(r, 10));
      concurrent--;
      return "ok";
    };

    const results = await Promise.all([
      limiter.run(task),
      limiter.run(task),
      limiter.run(task),
      limiter.run(task),
    ]);

    expect(results).toEqual(["ok", "ok", "ok", "ok"]);
    // At most 2 tasks ran concurrently
    expect(maxConcurrent).toBeLessThanOrEqual(2);
  });

  it("queues tasks beyond the limit and runs them later", async () => {
    const limiter = new ConcurrencyLimiter(1);

    // Use a controller to stagger tasks
    const order: number[] = [];
    const track = (id: number) => async () => {
      order.push(id);
      await new Promise((r) => setTimeout(r, 5));
      return id;
    };

    await Promise.all([
      limiter.run(track(1)),
      limiter.run(track(2)),
      limiter.run(track(3)),
    ]);

    // With limit 1 tasks must execute sequentially
    expect(order).toEqual([1, 2, 3]);
  });

  it("handles rejected tasks without breaking the limiter", async () => {
    const limiter = new ConcurrencyLimiter(2);

    // Submit a mix of failing and passing tasks
    const results = await Promise.allSettled([
      limiter.run(async () => {
        await new Promise((r) => setTimeout(r, 5));
        throw new Error("fail");
      }),
      limiter.run(async () => {
        await new Promise((r) => setTimeout(r, 5));
        return "ok1";
      }),
      limiter.run(async () => {
        await new Promise((r) => setTimeout(r, 5));
        return "ok2";
      }),
    ]);

    expect(results[0].status).toBe("rejected");
    expect(results[1].status).toBe("fulfilled");
    expect(results[2].status).toBe("fulfilled");
    if (results[1].status === "fulfilled") expect(results[1].value).toBe("ok1");
    if (results[2].status === "fulfilled") expect(results[2].value).toBe("ok2");

    // Limiter should still accept new tasks after a rejection
    const final = await limiter.run(async () => "still works");
    expect(final).toBe("still works");
  });

  it("respects limit of 1 (serial execution)", async () => {
    const limiter = new ConcurrencyLimiter(1);
    const order: number[] = [];

    await Promise.all([
      limiter.run(async () => {
        await new Promise((r) => setTimeout(r, 10));
        order.push(1);
      }),
      limiter.run(async () => {
        await new Promise((r) => setTimeout(r, 5));
        order.push(2);
      }),
    ]);

    // Even though task 2 has a shorter timeout, limit 1 means task 1
    // must start first, so task 1 finishes first.
    expect(order).toEqual([1, 2]);
  });
});