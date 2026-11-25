import * as React from "react";

const ELAPSED_REFRESH_INTERVAL_MS = 1_000;

/**
 * Returns the number of whole seconds elapsed since `startTimestamp`, ticking
 * once per second. Returns 0 while `startTimestamp` is empty or invalid.
 */
export function useElapsedSeconds(
  startTimestamp: string | null | undefined,
): number {
  const startMs = startTimestamp
    ? new Date(startTimestamp).getTime()
    : Number.NaN;

  const [elapsedSeconds, setElapsedSeconds] = React.useState(0);

  React.useEffect(() => {
    if (Number.isNaN(startMs)) {
      setElapsedSeconds(0);
      return undefined;
    }

    const update = () => {
      setElapsedSeconds(Math.max(0, Math.floor((Date.now() - startMs) / 1000)));
    };

    update();
    const timer = setInterval(update, ELAPSED_REFRESH_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [startMs]);

  return elapsedSeconds;
}
