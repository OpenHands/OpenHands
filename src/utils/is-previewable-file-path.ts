import { isMarkdownFilePath } from "./is-markdown-file-path";

/**
 * Extensions we render live inside a sandboxed iframe in the conversation:
 * markup formats a browser renders from a workspace URL, where relative asset
 * references must resolve.
 */
const FRAME_PREVIEW_EXTS = new Set(["html", "htm", "svg"]);

/** Raster images the browser can paint directly from a workspace URL. */
const IMAGE_PREVIEW_EXTS = new Set([
  "png",
  "jpg",
  "jpeg",
  "gif",
  "webp",
  "bmp",
  "ico",
  "avif",
]);

const PDF_PREVIEW_EXTS = new Set(["pdf"]);

/** OOXML packages we can unpack and outline without a renderer dependency. */
const OOXML_PREVIEW_EXTS = new Set(["docx", "xlsx", "pptx"]);

/**
 * How an artifact should be previewed inline, or `null` when it gets no rich
 * preview and stays on the plain CodeBlock / DiffView path.
 *
 * - `markdown` — rich markdown card (`MarkdownFilePreview`)
 * - `frame`    — sandboxed iframe pointed at the workspace fileserver
 * - `image`    — `<img>` pointed at the workspace fileserver
 * - `pdf`      — unsandboxed iframe so Chromium's PDF viewer can instantiate
 * - `ooxml`    — unpacked and outlined by `readOoxmlPreview`
 */
export type ArtifactPreviewKind =
  | "markdown"
  | "frame"
  | "image"
  | "pdf"
  | "ooxml";

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
  if (IMAGE_PREVIEW_EXTS.has(ext)) return "image";
  if (PDF_PREVIEW_EXTS.has(ext)) return "pdf";
  if (OOXML_PREVIEW_EXTS.has(ext)) return "ooxml";
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

/** True for `.docx` / `.xlsx` / `.pptx`. */
export function isOfficePreviewablePath(path: string): boolean {
  return OOXML_PREVIEW_EXTS.has(getFileExtension(path));
}

/**
 * True for any artifact path that gets a rich inline preview in the chat —
 * the markdown card, the sandboxed frame, a raster image, a PDF, or an
 * unpacked Office document.
 */
export function isPreviewableArtifactPath(path: string): boolean {
  return getArtifactPreviewKind(path) !== null;
}
