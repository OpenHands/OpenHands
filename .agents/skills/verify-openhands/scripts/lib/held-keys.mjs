// Keys held down across CLI calls (`browser key down|up|release-all`).
//
// Playwright's keyboard.down() on a key that is already down sends another
// keydown with `event.repeat === true`, which is how a held key reaches the
// page. A headless browser never auto-repeats, so repeats are explicit: a
// count and an interval. Each held key remembers the page it went down on,
// so switching tabs cannot strand it, and `key up` releases it there.
// Modifiers held this way also apply to later clicks and presses.

export const KEY_REPEAT_DEFAULTS = {
  intervalMs: 30,
  maxRepeat: 1000,
  // Keeps one command well inside the CLI's browser call timeout.
  maxTotalMs: 60_000,
};

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function keyError(message, hint) {
  return Object.assign(new Error(message), { hint });
}

export class HeldKeys {
  #keys = new Map();

  /** Key names in the order they went down. */
  get held() {
    return [...this.#keys.keys()];
  }

  /**
   * Press `key` and keep it down. A key that is already down gets repeat
   * keydowns only: `repeat` of them, or one when no count is given.
   */
  async down(page, key, { repeat, intervalMs } = {}) {
    const interval = intervalMs ?? KEY_REPEAT_DEFAULTS.intervalMs;
    const target = this.#keys.get(key) ?? page;
    let repeats = repeat ?? 0;
    if (this.#keys.has(key)) {
      repeats = repeat ?? 1;
    } else {
      await target.keyboard.down(key);
      this.#keys.set(key, target);
    }
    for (let i = 0; i < repeats; i += 1) {
      if (interval > 0) await sleep(interval);
      await target.keyboard.down(key);
    }
    return { key, repeats, held: this.held };
  }

  async up(key) {
    const page = this.#keys.get(key);
    if (!page) {
      throw keyError(
        `${key} is not held`,
        this.held.length
          ? `Held now: ${this.held.join(", ")}`
          : "Nothing is held; `browser key down <Key>` presses and holds a key.",
      );
    }
    this.#keys.delete(key);
    // A closed tab has already let go of its keys.
    await page.keyboard.up(key).catch(() => {});
    return { key, held: this.held };
  }

  /** Release every held key, last pressed first. */
  async releaseAll() {
    const released = this.held.reverse();
    for (const key of released) await this.up(key);
    return { released, held: this.held };
  }
}

/** Validate CLI flags for `browser key down`; returns daemon arguments. */
export function keyDownArguments(key, flags = {}) {
  const repeat = flags.repeat === undefined ? undefined : Number(flags.repeat);
  if (
    repeat !== undefined &&
    (!Number.isInteger(repeat) ||
      repeat < 0 ||
      repeat > KEY_REPEAT_DEFAULTS.maxRepeat)
  ) {
    throw keyError(
      `--repeat must be a whole number from 0 to ${KEY_REPEAT_DEFAULTS.maxRepeat}`,
      "control-openhands browser key down Enter --repeat 5 --interval-ms 50",
    );
  }
  const intervalMs =
    flags["interval-ms"] === undefined
      ? undefined
      : Number(flags["interval-ms"]);
  if (
    intervalMs !== undefined &&
    (!Number.isFinite(intervalMs) || intervalMs < 0 || intervalMs > 10_000)
  ) {
    throw keyError(
      "--interval-ms must be a number of milliseconds from 0 to 10000",
      "control-openhands browser key down Enter --repeat 5 --interval-ms 50",
    );
  }
  if (intervalMs !== undefined && repeat === undefined) {
    throw keyError(
      "--interval-ms only applies with --repeat",
      "control-openhands browser key down Enter --repeat 5 --interval-ms 50",
    );
  }
  const total = (repeat ?? 0) * (intervalMs ?? KEY_REPEAT_DEFAULTS.intervalMs);
  if (total > KEY_REPEAT_DEFAULTS.maxTotalMs) {
    throw keyError(
      `--repeat times --interval-ms is ${total} ms; keep it within ${KEY_REPEAT_DEFAULTS.maxTotalMs} ms`,
      "Split a long hold into several `browser key down <Key> --repeat N` calls.",
    );
  }
  return { key, repeat, intervalMs };
}
