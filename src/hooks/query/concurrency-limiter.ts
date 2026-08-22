/**
 * Simple FIFO concurrency limiter for async work. Keeps at most `max` tasks
 * running at once; additional tasks are queued and start as soon as a slot
 * frees up.
 *
 * Used to bound request fan-out that would otherwise saturate the browser's
 * per-origin connection pool and starve unrelated queries (e.g. the
 * conversation list while many automation run-history requests are in flight).
 */
export class ConcurrencyLimiter {
  private active = 0;

  private readonly queue: Array<() => void> = [];

  constructor(private readonly max: number) {}

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
      this.active += 1;
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      this.queue.push(resolve);
    });
  }

  private release(): void {
    const next = this.queue.shift();
    if (next) {
      next();
    } else {
      this.active -= 1;
    }
  }
}

/**
 * Shared limiter for automation run-history requests.
 *
 * The home page and automations list page fan out one `getAutomationRuns`
 * request per automation. Left unbounded, a large number of enabled
 * automations can consume the browser's connection pool and prevent other
 * requests (such as the conversation list `searchConversations`) from making
 * progress while an automation run is keeping those requests slow.
 */
export const automationRunRequestsLimiter = new ConcurrencyLimiter(3);
