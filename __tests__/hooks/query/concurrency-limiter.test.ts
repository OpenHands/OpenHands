import { describe, expect, it } from "vitest";
import { ConcurrencyLimiter } from "#/hooks/query/concurrency-limiter";

/** A task whose completion the test controls. */
function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

async function flushMicrotasks() {
  await Promise.resolve();
  await Promise.resolve();
}

describe("ConcurrencyLimiter", () => {
  it("runs at most `limit` tasks at once and starts queued tasks in FIFO order as slots free", async () => {
    // Arrange — five controllable tasks behind a limit of two.
    const limiter = new ConcurrencyLimiter(2);
    const started: number[] = [];
    const gates = [0, 1, 2, 3, 4].map(() => deferred<void>());
    const runs = gates.map((gate, index) =>
      limiter.run(() => {
        started.push(index);
        return gate.promise;
      }),
    );

    // Act & Assert — only the first two start; each completion admits the
    // next waiter in arrival order.
    await flushMicrotasks();
    expect(started).toEqual([0, 1]);

    gates[0].resolve();
    await flushMicrotasks();
    expect(started).toEqual([0, 1, 2]);

    gates[1].resolve();
    gates[2].resolve();
    await flushMicrotasks();
    expect(started).toEqual([0, 1, 2, 3, 4]);

    gates[3].resolve();
    gates[4].resolve();
    await expect(Promise.all(runs)).resolves.toBeDefined();
  });

  it("releases a slot when a task rejects, so later tasks still run", async () => {
    // Arrange — a single slot occupied by a failing task.
    const limiter = new ConcurrencyLimiter(1);

    // Act
    const failing = limiter.run(() => Promise.reject(new Error("boom")));

    // Assert — the failure propagates and the slot is free for the next task.
    await expect(failing).rejects.toThrow("boom");
    await expect(limiter.run(() => Promise.resolve("next"))).resolves.toBe(
      "next",
    );
  });

  it("rejects a task whose signal aborted while queued, without running it", async () => {
    // Arrange — one slot held open so the second task queues, then gets
    // aborted before a slot frees (a query unmounting while it waits).
    const limiter = new ConcurrencyLimiter(1);
    const gate = deferred<void>();
    const first = limiter.run(() => gate.promise);
    const controller = new AbortController();
    let queuedTaskRan = false;
    const queued = limiter.run(() => {
      queuedTaskRan = true;
      return Promise.resolve("ran");
    }, controller.signal);
    queued.catch(() => {}); // observed below; avoid an unhandled rejection

    // Act
    controller.abort();
    gate.resolve();

    // Assert — the aborted task rejects without executing, and its slot is
    // handed on to later work.
    await expect(queued).rejects.toThrow();
    expect(queuedTaskRan).toBe(false);
    await expect(limiter.run(() => Promise.resolve("after"))).resolves.toBe(
      "after",
    );
    await first;
  });
});
