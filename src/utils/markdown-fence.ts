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
 * backtick longer than any run inside ``content``. Content that starts or
 * ends with a backtick is padded with a space (which CommonMark strips) so it
 * can't merge with the delimiter. Content that both starts and ends with a space
 * (and does not consist entirely of spaces) is also padded so CommonMark's
 * strip rule does not remove its leading and trailing spaces.
 */
export const markdownInlineCode = (content: string): string => {
  const ticks = "`".repeat(longestBacktickRun(content) + 1);
  const edgeSpaces =
    content.startsWith(" ") && content.endsWith(" ") && /[^ ]/.test(content);
  const padded = /^`|`$/.test(content) || edgeSpaces ? ` ${content} ` : content;
  return `${ticks}${padded}${ticks}`;
};
