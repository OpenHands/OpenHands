import type { Plugin } from "unified";
import type { ElementContent, Root } from "hast";
import { SKIP, visit } from "unist-util-visit";

/** Must match the `stream-word-fade` animation in index.css. */
export const STREAM_WORD_FADE_MS = 280;

/** Text inside these is not split into words. */
const SKIP_TAGS = new Set(["pre", "code", "svg", "math", "script", "style"]);

/**
 * Records when each part of a streamed text was revealed, so a word's fade can
 * be resumed where it was rather than restarted when its element is re-created
 * (a streamed block is re-mounted when the block before it is frozen).
 */
export class RevealClock {
  private segments: { start: number; at: number }[] = [];

  private length = 0;

  /** The text now shown is `length` long. */
  advance(length: number, at = performance.now()) {
    if (length > this.length) {
      this.segments.push({ start: this.length, at });
      this.length = length;
    }
    // Segments whose fade is over no longer need a time.
    while (
      this.segments.length > 0 &&
      at - this.segments[0].at >= STREAM_WORD_FADE_MS
    ) {
      this.segments.shift();
    }
  }

  /** Treat text up to `length` as already shown, e.g. after a retry snapped in. */
  reset(length: number) {
    this.segments = [];
    this.length = length;
  }

  /** When the character at `offset` was revealed, or null if long enough ago. */
  revealedAt(offset: number): number | null {
    for (let i = this.segments.length - 1; i >= 0; i -= 1) {
      if (this.segments[i].start <= offset) {
        return this.segments[i].at;
      }
    }
    return null;
  }
}

interface FadeInWordsOptions {
  clock: RevealClock;
  /** Offset of this block's source within the whole streamed text. */
  blockStart: number;
}

/**
 * Wrap each word of a streamed reply in a span, and fade in the ones revealed
 * within the last `STREAM_WORD_FADE_MS`. Every word keeps a span (not only the
 * fading ones) so elements stay put as text is appended; a negative
 * `animation-delay` places a re-created word at its current point in the fade.
 *
 * Text nodes after inline markup carry no source position, so their offsets
 * continue from the previous one — off by the markup's few characters, which
 * is invisible in the timing.
 */
export const rehypeFadeInWords: Plugin<[FadeInWordsOptions], Root> =
  ({ clock, blockStart }) =>
  (tree) => {
    const renderedAt = performance.now();
    let cursor = 0;

    const wordSpan = (word: string, offset: number): ElementContent => {
      const revealedAt = clock.revealedAt(blockStart + offset);
      const elapsed = revealedAt === null ? Infinity : renderedAt - revealedAt;
      return {
        type: "element",
        tagName: "span",
        properties:
          elapsed < STREAM_WORD_FADE_MS
            ? {
                dataStreamWord: true,
                className: ["stream-word-fade"],
                style: `animation-delay:${-Math.round(elapsed)}ms`,
              }
            : { dataStreamWord: true },
        children: [{ type: "text", value: word }],
      };
    };

    visit(tree, (node, index, parent) => {
      if (node.type === "element" && SKIP_TAGS.has(node.tagName)) {
        return SKIP;
      }
      if (node.type !== "text" || !parent || index === undefined) {
        return undefined;
      }
      cursor = node.position?.start.offset ?? cursor;
      const parts: ElementContent[] = [];
      let offset = cursor;
      for (const part of node.value.split(/(\s+)/)) {
        if (part) {
          parts.push(
            /\S/.test(part)
              ? wordSpan(part, offset)
              : { type: "text", value: part },
          );
        }
        offset += part.length;
      }
      cursor += node.value.length;
      parent.children.splice(index, 1, ...parts);
      // Continue after the inserted nodes: their text is already split.
      return index + parts.length;
    });
  };
