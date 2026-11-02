const longestBacktickRun = (content: string): number =>
  Math.max(
    0,
    ...Array.from(content.matchAll(/`+/g), (match) => match[0].length),
  );

/**
 * Wrap ``content`` in a fenced code block whose fence is longer than any
 * backtick run inside it, so content that carries its own fences (a README,
 * a markdown-escaped tool result) can't close the block early and spill out
 * as live markdown.
 */
export const markdownFence = (content: string, language = ""): string => {
  const fence = "`".repeat(Math.max(3, longestBacktickRun(content) + 1));
  return `${fence}${language}\n${content}\n${fence}`;
};

/**
 * Inline-code counterpart of {@link markdownFence}: the delimiter is one
 * backtick longer than any run inside ``content``, and content that starts or
 * ends with a backtick is padded with a space (which CommonMark strips) so it
 * can't merge with the delimiter.
 */
export const markdownInlineCode = (content: string): string => {
  const ticks = "`".repeat(longestBacktickRun(content) + 1);
  const padded = /^`|`$/.test(content) ? ` ${content} ` : content;
  return `${ticks}${padded}${ticks}`;
};
