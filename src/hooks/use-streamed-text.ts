import React from "react";
import { create } from "zustand";

/** Reveal pace, in characters per millisecond (60 chars/s). */
const PACE = 0.06;

/**
 * Backlog is drained at `backlog / MAX_LAG_MS` per ms once that exceeds
 * `PACE`, so the reveal trails a fast model by about this long and catches
 * up within about a second of it stopping.
 */
const MAX_LAG_MS = 500;

/**
 * Hooks still revealing text, so the chat can keep following a reply that is
 * still typing out after its stream has already finished.
 */
const useRevealingStore = create<{ ids: ReadonlySet<symbol> }>(() => ({
  ids: new Set(),
}));

const setRevealing = (id: symbol, isRevealing: boolean) =>
  useRevealingStore.setState((state) => {
    if (state.ids.has(id) === isRevealing) {
      return state;
    }
    const ids = new Set(state.ids);
    if (isRevealing) {
      ids.add(id);
    } else {
      ids.delete(id);
    }
    return { ids };
  });

/** Whether any streamed text is still being revealed on screen. */
export const useIsRevealingStreamedText = () =>
  useRevealingStore((state) => state.ids.size > 0);

/**
 * Reveal `target` on a clock rather than at the granularity the network
 * happened to deliver it (#15493).
 *
 * Text is revealed at a steady `PACE`, so chunks that land in bursts read as
 * writing instead of a jump followed by a pause; only a backlog worth more
 * than `MAX_LAG_MS` speeds it up. Text that is not an extension of what is
 * already on screen (a superseding retry, or a different slot) snaps
 * immediately rather than being re-typed.
 */
export function useStreamedText(target: string): string {
  const [revealed, setRevealed] = React.useState(target);
  const revealedRef = React.useRef(target);
  // Fractional reveal position: the pace is under one character per frame.
  const positionRef = React.useRef(target.length);
  const frameRef = React.useRef<number | null>(null);
  const [revealId] = React.useState(() => Symbol("streamed-text"));

  React.useEffect(() => {
    const commit = (next: string) => {
      revealedRef.current = next;
      positionRef.current = next.length;
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

      const backlog = target.length - positionRef.current;
      const step = Math.max(PACE, backlog / MAX_LAG_MS) * elapsed;
      const position = Math.min(target.length, positionRef.current + step);
      const nextLength = Math.floor(position);
      if (nextLength !== revealedRef.current.length) {
        commit(target.slice(0, nextLength));
      }
      positionRef.current = position;

      if (nextLength < target.length) {
        frameRef.current = requestAnimationFrame(tick);
      } else {
        setRevealing(revealId, false);
      }
    };

    setRevealing(revealId, true);
    frameRef.current = requestAnimationFrame(tick);
    return () => {
      setRevealing(revealId, false);
      if (frameRef.current !== null) {
        cancelAnimationFrame(frameRef.current);
        frameRef.current = null;
      }
    };
  }, [target, revealId]);

  return revealed;
}
