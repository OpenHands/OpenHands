import React from "react";

/**
 * Window over which the unrevealed backlog is drained. Short enough that the
 * text never visibly trails the model, long enough that a burst of tokens
 * reads as typing rather than as a paste.
 */
const DRAIN_WINDOW_MS = 120;

/**
 * Reveal `target` on a clock rather than at the granularity the network
 * happened to deliver it (#15493).
 *
 * The rate is proportional to the backlog, so a fast model is revealed fast
 * and a slow one still reads smoothly; text that is not an extension of what
 * is already on screen (a superseding retry, or a different slot) snaps
 * immediately rather than being re-typed.
 */
export function useStreamedText(target: string): string {
  const [revealed, setRevealed] = React.useState(target);
  const revealedRef = React.useRef(target);
  const frameRef = React.useRef<number | null>(null);

  React.useEffect(() => {
    const commit = (next: string) => {
      revealedRef.current = next;
      setRevealed(next);
    };

    if (
      !target.startsWith(revealedRef.current) ||
      typeof requestAnimationFrame !== "function"
    ) {
      commit(target);
      return undefined;
    }
    if (revealedRef.current.length === target.length) {
      return undefined;
    }

    let lastTick = 0;
    const tick = (now: number) => {
      frameRef.current = null;
      const elapsed = lastTick ? now - lastTick : 16;
      lastTick = now;

      const backlog = target.length - revealedRef.current.length;
      const step = Math.max(
        1,
        Math.ceil((backlog * elapsed) / DRAIN_WINDOW_MS),
      );
      const nextLength = Math.min(
        target.length,
        revealedRef.current.length + step,
      );
      commit(target.slice(0, nextLength));

      if (nextLength < target.length) {
        frameRef.current = requestAnimationFrame(tick);
      }
    };

    frameRef.current = requestAnimationFrame(tick);
    return () => {
      if (frameRef.current !== null) {
        cancelAnimationFrame(frameRef.current);
        frameRef.current = null;
      }
    };
  }, [target]);

  return revealed;
}
