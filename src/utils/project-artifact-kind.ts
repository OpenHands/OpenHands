/**
 * How Canvas shows a project file, decided from its name alone: the server
 * sends every file as `application/octet-stream`.
 */
export type ProjectArtifactKind = "text" | "image" | "pdf" | "other";

const TEXT =
  /\.(md|markdown|txt|json|csv|log|ya?ml|toml|diff|patch|ts|tsx|js|py|html|css|sh)$/i;

const IMAGE_TYPES: Record<string, string> = {
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  gif: "image/gif",
  webp: "image/webp",
  avif: "image/avif",
  bmp: "image/bmp",
  svg: "image/svg+xml",
};

const extension = (name: string) =>
  name.includes(".") ? name.split(".").pop()!.toLowerCase() : "";

export function projectArtifactKind(name: string): ProjectArtifactKind {
  if (TEXT.test(name)) return "text";
  const ext = extension(name);
  if (ext in IMAGE_TYPES) return "image";
  if (ext === "pdf") return "pdf";
  return "other";
}

/**
 * The media type a downloaded file gets. Only images and PDFs get their own
 * type, because only they are rendered; an `.html` file stays plain text so
 * it never runs from Canvas's origin.
 */
export function artifactMediaType(name: string): string {
  const ext = extension(name);
  if (ext in IMAGE_TYPES) return IMAGE_TYPES[ext];
  if (ext === "pdf") return "application/pdf";
  if (TEXT.test(name)) return "text/plain";
  return "application/octet-stream";
}
