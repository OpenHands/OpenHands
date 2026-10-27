import { fromMarkdown } from "mdast-util-from-markdown";

/**
 * Advance the end of the markdown prefix that can no longer change as text is
 * appended, parsing only what lies after `from`.
 *
 * A top-level block is stable once the block after it, across a blank line, is
 * itself complete (another block follows it). Until then the next block can
 * still merge back: in `1. a\n\n2` the `2` becomes the list's second item once
 * the `.` arrives, which also turns the whole list loose. A streamed reply can
 * then render its finished blocks once and re-parse only the tail, instead of
 * the whole message every frame.
 *
 * Raw HTML (a `<details>` spanning blank lines) and link definitions (which
 * resolve references elsewhere) cannot be read block by block, so the boundary
 * stops before them and the rest stays in the tail. The finished message is
 * rendered in one piece, so this only affects the reply while it streams.
 */
export function advanceStableMarkdownBoundary(
  text: string,
  from: number,
): number {
  const tail = text.slice(from);
  const blocks = fromMarkdown(tail).children;
  let boundary = from;

  for (let i = 0; i < blocks.length - 2; i += 1) {
    const block = blocks[i];
    if (block.type === "html" || block.type === "definition") {
      break;
    }
    const end = block.position?.end.offset;
    const nextStart = blocks[i + 1].position?.start.offset;
    if (
      end === undefined ||
      nextStart === undefined ||
      !/\n[ \t]*\n/.test(tail.slice(end, nextStart))
    ) {
      break;
    }
    boundary = from + nextStart;
  }

  return boundary;
}
