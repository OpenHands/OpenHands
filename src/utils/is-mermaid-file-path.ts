/** Extensions whose contents are Mermaid diagram source. */
const MERMAID_EXTS = new Set(["mmd", "mermaid"]);

/**
 * True when `path` ends with a Mermaid extension (`.mmd`, `.mermaid`).
 */
export function isMermaidFilePath(path: string): boolean {
  const idx = path.lastIndexOf(".");
  if (idx === -1) return false;
  return MERMAID_EXTS.has(path.slice(idx + 1).toLowerCase());
}
