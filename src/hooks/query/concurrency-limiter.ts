/**
 * A simple concurrency limiter (semaphore) that bounds the number of
 * simultaneously-executing async tasks. Tasks beyond the limit are queued
 * and dispatched as earlier ones complete.
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
      next();
    } else {
      this.active--;
    }
  }
}

/** Max concurrent per-automation run-history requests (automation DB pool ~15). */
export const automationRunRequestsLimiter = new ConcurrencyLimiter(3);
