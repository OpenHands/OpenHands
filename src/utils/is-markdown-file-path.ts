/** Extensions treated as Markdown for rich preview rendering. */
const MARKDOWN_EXTS = new Set(["md", "markdown", "mdx"]);

/**
 * True when `path` ends with a Markdown extension (`.md`, `.markdown`, `.mdx`).
 */
export function isMarkdownFilePath(path: string): boolean {
  const idx = path.lastIndexOf(".");
  if (idx === -1) return false;
  return MARKDOWN_EXTS.has(path.slice(idx + 1).toLowerCase());
}
