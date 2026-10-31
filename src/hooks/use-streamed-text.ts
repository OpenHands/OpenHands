import React from "react";
import { create } from "zustand";

/** Smooth bursts without holding received text for more than 100ms. */
const MAX_REVEAL_MS = 100;

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

/** Smooth live deltas briefly; completed replies are immediately readable. */
export function useStreamedText(target: string, streaming = true): string {
  const [revealed, setRevealed] = React.useState(target);
  const revealedRef = React.useRef(target);
  // Keep fractional positions when a short chunk advances less than one character.
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
      !streaming ||
      !target.startsWith(revealedRef.current) ||
      typeof requestAnimationFrame !== "function"
    ) {
      commit(target);
      return undefined;
    }
    if (revealedRef.current.length === target.length) {
      return undefined;
    }

    const speed = (target.length - positionRef.current) / MAX_REVEAL_MS;
    let lastTick = 0;
    const tick = (now: number) => {
      frameRef.current = null;
      const elapsed = lastTick ? now - lastTick : 16;
      lastTick = now;

      const step = speed * elapsed;
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
  }, [target, revealId, streaming]);

  return streaming ? revealed : target;
}
