/**
 * A simple concurrency limiter (semaphore) that bounds the number of
 * simultaneously-executing async tasks.  Tasks beyond the limit are queued
 * and dispatched as earlier ones complete.
 *
 * Unlike a general-purpose semaphore this type is deliberately small and
 * non-configurable at runtime — the limit is set at construction time.
 *
 * @example
 * ```ts
 * const limiter = new ConcurrencyLimiter(3);
 * await limiter.run(() => fetch("/api/slow"));
 * ```
 */
export class ConcurrencyLimiter {
  private active = 0;
  private readonly queue: Array<() => void> = [];

  constructor(private readonly max: number) {}

  /**
   * Run `task` under the concurrency limit.  Returns the task's result.
   * If the limit is already reached, the task is queued until a running
   * task completes.
   */
  async run<T>(task: () => Promise<T>): Promise<T> {
    await this.acquire();
    try {
      return await task();
    } finally {
      this.release();
    }
  }

  private acquire(): Promise<void> {
    if (this.active < this.max) {
      this.active++;
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      this.queue.push(resolve);
    });
  }

  private release(): void {
    const next = this.queue.shift();
    if (next) {
      // Transfer the slot of the completing task directly to the next waiter.
      next();
    } else {
      this.active--;
    }
  }
}

/**
 * Shared concurrency limiter for per-automation run-history requests.
 *
 * The automation service's database connection pool is small (default pool
 * size 10 + overflow 5, pool timeout 30 s).  Firing one request per listed
 * automation (up to 50) in parallel exhausts it and causes ~30 s request
 * timeouts.  Three concurrent requests keeps the service well within its
 * pool.
 *
 * This singleton is imported by both useAutomationRunSummaries and
 * useLatestAutomationRuns, so the total concurrent request rate across
 * both surfaces stays bounded.
 */
export const automationRunRequestsLimiter = new ConcurrencyLimiter(3);