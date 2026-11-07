/**
 * A FIFO promise semaphore: at most `limit` wrapped tasks run at once; the
 * rest wait in arrival order. Wrap a queryFn's work in `run` to bound how many
 * of a query family's requests are in flight simultaneously.
 */
export class ConcurrencyLimiter {
  private active = 0;

  private readonly queue: Array<() => void> = [];

  constructor(private readonly limit: number) {}

  async run<T>(task: () => Promise<T>, signal?: AbortSignal): Promise<T> {
    await this.acquire();
    try {
      // A query cancelled while queued (e.g. its last observer unmounted)
      // frees its slot without ever hitting the network.
      if (signal?.aborted) {
        throw signal.reason ?? new DOMException("Aborted", "AbortError");
      }
      return await task();
    } finally {
      this.release();
    }
  }

  private acquire(): Promise<void> {
    if (this.active < this.limit) {
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
      // The waiter inherits the released slot directly, so `active` stays put.
      next();
    } else {
      this.active -= 1;
    }
  }
}

/**
 * Shared by every per-automation runs query: the dashboard and home page each
 * fan out one GET .../runs request per automation, which can exhaust the
 * automation service's DB connection pool (~15 connections) and stall every
 * request for the pool's 30s acquisition timeout. Three in flight keeps the
 * surfaces responsive without saturating the pool.
 */
export const automationRunRequestsLimiter = new ConcurrencyLimiter(3);
