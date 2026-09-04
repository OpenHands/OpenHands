import { useEffect, useState } from "react";

/**
 * Returns the current wall-clock time in milliseconds, updated every second
 * while `active` is true.
 *
 * The initial value is captured once inside the `useState` initializer (not
 * during render) and subsequent values are set inside the interval callback
 * (inside `useEffect`), so no impure call occurs at render time.
 *
 * Cleans up the interval automatically when `active` becomes false or on
 * component unmount so there are no lingering timers after completion.
 */
export function useSecondsTick(active: boolean): number {
  const [nowMs, setNowMs] = useState(() => Date.now());

  useEffect(() => {
    if (!active) return undefined;
    const timer = setInterval(() => setNowMs(Date.now()), 1_000);
    return () => clearInterval(timer);
  }, [active]);

  return nowMs;
}
