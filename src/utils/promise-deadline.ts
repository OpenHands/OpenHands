export class PromiseDeadlineError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PromiseDeadlineError";
  }
}

/**
 * Settle a promise within a caller-owned deadline.
 *
 * The underlying operation is still observed after the deadline wins, so a
 * later rejection cannot become unhandled. The timer is cleared whenever the
 * operation settles first.
 */
export function withPromiseDeadline<T>(
  operation: PromiseLike<T>,
  deadlineMs: number,
  timeoutMessage: string,
): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    let settled = false;
    const timeoutId = setTimeout(() => {
      if (settled) return;
      settled = true;
      reject(new PromiseDeadlineError(timeoutMessage));
    }, deadlineMs);

    Promise.resolve(operation).then(
      (value) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeoutId);
        resolve(value);
      },
      (error: unknown) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeoutId);
        reject(error);
      },
    );
  });
}
