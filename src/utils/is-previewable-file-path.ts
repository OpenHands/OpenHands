import { isMarkdownFilePath } from "./is-markdown-file-path";

/**
 * Extensions we render live inside a sandboxed iframe in the conversation:
 * markup formats a browser renders from a workspace URL, where relative asset
 * references must resolve.
 *
 * This allowlist is deliberately narrow. Binary formats — raster images and
 * PDFs — and Office documents (`.docx` / `.xlsx` / `.pptx`) are intentionally
 * **not** previewed inline and keep their existing CodeBlock / fallback
 * rendering. Binary previews are an explicit non-goal of #18113, and a PDF
 * needs an unsandboxed frame, so it would auto-mount an untrusted document load
 * for every create event.
 */
const FRAME_PREVIEW_EXTS = new Set(["html", "htm", "svg"]);

/**
 * How an artifact should be previewed inline, or `null` when it gets no rich
 * preview and stays on the plain CodeBlock / DiffView path.
 *
 * - `markdown` — rich markdown card (`MarkdownFilePreview`)
 * - `frame`    — sandboxed iframe pointed at the workspace fileserver
 */
export type ArtifactPreviewKind = "markdown" | "frame";

export function getFileExtension(path: string): string {
  const idx = path.lastIndexOf(".");
  if (idx === -1) return "";
  return path.slice(idx + 1).toLowerCase();
}

export function getArtifactPreviewKind(
  path: string,
): ArtifactPreviewKind | null {
  if (isMarkdownFilePath(path)) return "markdown";
  const ext = getFileExtension(path);
  if (FRAME_PREVIEW_EXTS.has(ext)) return "frame";
  return null;
}

/**
 * True when `path` is HTML/SVG, i.e. renderable in a sandboxed iframe.
 *
 * `.svg` is intentionally treated as a frame preview here even though the
 * workspace file hook also classifies it as an image — the inline card renders
 * it via `<iframe>` so relative references inside the SVG resolve.
 */
export function isFramePreviewablePath(path: string): boolean {
  return FRAME_PREVIEW_EXTS.has(getFileExtension(path));
}

/**
 * True for any artifact path that gets a rich inline preview in the chat —
 * the markdown card or the sandboxed frame.
 */
export function isPreviewableArtifactPath(path: string): boolean {
  return getArtifactPreviewKind(path) !== null;
}
