/**
 * Wrap ``content`` in a fenced code block whose fence is longer than any
 * backtick run inside it, so content that carries its own fences (a README,
 * a markdown-escaped tool result) can't close the block early and spill out
 * as live markdown.
 */
export const markdownFence = (content: string, language = ""): string => {
  const longestRun = Math.max(
    0,
    ...Array.from(content.matchAll(/`+/g), (match) => match[0].length),
  );
  const fence = "`".repeat(Math.max(3, longestRun + 1));
  return `${fence}${language}\n${content}\n${fence}`;
};
